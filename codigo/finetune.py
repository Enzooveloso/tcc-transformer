"""Fine-tuning de recuperação pós-poda — o Estágio 1 da pipeline.

Todos os resultados anteriores da pipeline são *one-shot*: poda-se o modelo
pré-treinado e mede-se imediatamente. Esse regime isola o dano causado pela
poda, mas não é o que a literatura faz: tanto \\citet{Han2015} quanto
\\citet{Li2017} retreinam o modelo depois de podá-lo, e é o retreinamento que
recupera boa parte da qualidade perdida. Este módulo fecha essa lacuna.

O que ele mede, para cada configuração podada:

  - ``perplexity_pre``  — a perplexity one-shot, logo após a poda;
  - ``perplexity``      — a perplexity depois do fine-tuning;
  - ``recuperacao_log`` — quanto da degradação foi revertida, em escala log.

**Controle denso (``--estrategia nenhuma``): leia isto antes de interpretar
qualquer resultado.** O GPT-2 não foi pré-treinado no WikiText-2, de modo que
o simples fine-tuning do modelo DENSO nesse corpus já reduz a perplexity por
adaptação de domínio, sem poda nenhuma envolvida. Sem essa linha de controle é
impossível separar "recuperei o dano da poda" de "adaptei o modelo ao corpus":
toda queda de perplexity observada nas configurações podadas precisa ser lida
contra a queda que o modelo denso obtém com o mesmo orçamento de treino.
Rode sempre ``--estrategia nenhuma`` na mesma sessão que as demais.

**Preservação da esparsidade.** Na poda não estruturada os pesos são apenas
zerados: a arquitetura continua íntegra e, sem cuidado, o gradiente ressuscita
os pesos podados já no primeiro passo do otimizador — ao fim do treino o
modelo estaria denso de novo. Como em \\citet{Han2015}, mantém-se uma máscara
binária fixa, capturada logo após a poda e reaplicada depois de cada passo do
otimizador. A coluna ``esparsidade_pos`` registra a esparsidade medida ao
final do treino e serve de verificação: ela deve coincidir com
``esparsidade_real``. Na poda estruturada não há máscara — as estruturas foram
fisicamente removidas e o que sobrou é uma arquitetura menor e densa.

**Determinismo.** Ao contrário do resto da pipeline, o fine-tuning é
estocástico (ordem dos lotes, dropout). A semente passa a importar de fato
aqui, e repetições com sementes distintas passam a fazer sentido — o que não
acontecia nas varreduras one-shot, todas determinísticas.

Uso como script:
    python finetune.py --estrategia nenhuma                  # controle denso
    python finetune.py --estrategia magnitude --scope uniforme
    python finetune.py --estrategia estruturada --scope global
    python finetune.py --estrategia sensibilidade --beta 1.0
    python finetune.py --estrategia magnitude --sparsities 0.3 0.5 --epochs 1
    python finetune.py --estrategia nenhuma --max-steps 50 --eval-every 25
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from data import load_encodings
from energy import track_energy
from eval import compute_perplexity, evaluate_all
from model import load_model_and_tokenizer
from prune_magnitude import (
    _prunable_named_weights,
    prune_magnitude,
    prunable_sparsity,
)
from prune_structured import prune_structured, prunable_param_count
from sensitivity import (
    allocate_rates,
    apply_rates,
    load_profile,
    _layer_param_weights,
)
from utils import append_result, set_seed

ESTRATEGIAS = ("nenhuma", "magnitude", "estruturada", "sensibilidade")

RESULTADOS_CSV = "finetune.csv"
CURVA_CSV = "finetune_curva.csv"


# ---------------------------------------------------------------------------
# Dados de treino
# ---------------------------------------------------------------------------

def load_blocks(cfg: Config, tokenizer, split: str, block_size: int,
                limite: int | None = None) -> torch.Tensor:
    """Tokeniza um split do corpus e o fatia em blocos contíguos de treino.

    A avaliação de perplexity usa janelas deslizantes com sobreposição
    (``eval.compute_perplexity``), estratégia adequada para medir. Para treinar,
    o padrão é outro: o fluxo de tokens é cortado em blocos disjuntos de
    ``block_size``, cada um uma amostra independente. O resto final, menor que
    um bloco, é descartado.

    Devolve um tensor (n_blocos, block_size) em CPU — os lotes vão para a GPU
    um a um, no laço de treino.
    """
    ids = load_encodings(replace(cfg, dataset_split=split), tokenizer)[0]
    n_blocos = ids.size(0) // block_size
    if n_blocos == 0:
        raise ValueError(
            f"split {split!r} tem {ids.size(0)} tokens, menos que um bloco "
            f"de {block_size}"
        )
    blocos = ids[: n_blocos * block_size].view(n_blocos, block_size)
    if limite is not None:
        blocos = blocos[:limite]
    return blocos


# ---------------------------------------------------------------------------
# Máscaras: preservação da esparsidade durante o treino
# ---------------------------------------------------------------------------

def capture_masks(model) -> dict[str, torch.Tensor]:
    """Fotografa o padrão de zeros das matrizes podáveis, logo após a poda.

    A máscara é ``w != 0`` sobre exatamente as mesmas matrizes que a poda por
    magnitude considera podáveis. Pesos que já fossem nulos no modelo
    pré-treinado (raríssimos) entram na máscara e ficam congelados junto; o
    efeito sobre o resultado é desprezível e a alternativa — comparar contra
    uma cópia do modelo denso — custaria o dobro de memória.

    Devolve um dicionário {nome do módulo: máscara booleana}. Chamar em um
    modelo não podado devolve máscaras cheias de ``True``, o que torna a
    reaplicação um no-op seguro.
    """
    return {nome: (w != 0) for nome, w in _prunable_named_weights(model)}


@torch.no_grad()
def reapply_masks(model, masks: dict[str, torch.Tensor]) -> None:
    """Rezera os pesos podados. Deve rodar depois de CADA passo do otimizador.

    Zerar o gradiente não basta: o AdamW tem momento e *weight decay*, e ambos
    alteram um peso mesmo quando seu gradiente é nulo. Reaplicar a máscara
    depois do passo é o que efetivamente mantém a esparsidade.
    """
    for nome, w in _prunable_named_weights(model):
        mask = masks.get(nome)
        if mask is not None:
            w.mul_(mask.to(w.dtype))


# ---------------------------------------------------------------------------
# Poda: despacho para as três estratégias já implementadas
# ---------------------------------------------------------------------------

def apply_strategy(cfg: Config, model, estrategia: str, sparsity: float,
                   scope: str, alvo: str, beta: float, sens_base: str) -> dict:
    """Poda ``model`` in-place segundo a estratégia pedida.

    Reaproveita integralmente os módulos das varreduras one-shot, para que o
    modelo treinado aqui seja o MESMO que aparece no capítulo de Resultados sem
    fine-tuning — a única diferença entre as duas linhas passa a ser o treino.

    Devolve os campos descritivos da configuração (para o CSV).
    """
    if estrategia == "nenhuma" or sparsity <= 0:
        return {
            "escopo": "-" if estrategia == "nenhuma" else scope,
            "alvo": "-",
            "esparsidade_real": 0.0,
            "cabecas_removidas": 0,
            "neuronios_removidos": 0,
            "taxas_por_camada": "",
        }

    if estrategia == "magnitude":
        prune_magnitude(model, sparsity, scope=scope)
        return {
            "escopo": scope,
            "alvo": "-",
            "esparsidade_real": prunable_sparsity(model),
            "cabecas_removidas": 0,
            "neuronios_removidos": 0,
            "taxas_por_camada": "",
        }

    if estrategia == "estruturada":
        antes = prunable_param_count(model)
        prune_structured(model, sparsity, scope=scope, alvo=alvo)
        return {
            "escopo": scope,
            "alvo": alvo,
            "esparsidade_real": 1.0 - prunable_param_count(model) / antes,
            # As contagens exatas exigiriam instrumentar prune_structured; o
            # que importa para o CSV é a fração de parâmetros, já registrada.
            "cabecas_removidas": -1,
            "neuronios_removidos": -1,
            "taxas_por_camada": "",
        }

    if estrategia == "sensibilidade":
        # Qual perfil guia a alocação. Perfis de 'magnitude' e 'estruturada'
        # são levantados separadamente e NÃO são intercambiáveis: a correlação
        # de postos entre eles é de apenas 0,50 (ver o capítulo de Resultados),
        # de modo que usar um no lugar do outro alocaria o orçamento com um
        # mapa que pertence a outro método. Por isso é escolha explícita.
        base = sens_base
        deltas, _ = load_profile(cfg, base)
        pesos = _layer_param_weights(model)
        rates = allocate_rates(deltas, pesos, sparsity, beta=beta)

        antes = prunable_param_count(model)
        cabecas, neuronios = apply_rates(model, base, rates)
        real = (prunable_sparsity(model) if base == "magnitude"
                else 1.0 - prunable_param_count(model) / antes)
        return {
            "escopo": f"sensibilidade:{base}",
            "alvo": alvo if base == "estruturada" else "-",
            "esparsidade_real": real,
            "cabecas_removidas": cabecas,
            "neuronios_removidos": neuronios,
            "taxas_por_camada": ";".join(f"{r:.4f}" for r in rates),
        }

    raise ValueError(f"estrategia inválida: {estrategia!r} (use {ESTRATEGIAS})")


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def _param_groups(model, weight_decay: float) -> list[dict]:
    """Separa os parâmetros que recebem *weight decay* dos que não recebem.

    Convenção padrão em Transformers: vieses e ganhos de LayerNorm são
    parâmetros de escala/deslocamento, e penalizá-los prejudica a normalização
    sem trazer regularização útil.
    """
    decay, no_decay = [], []
    for nome, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or nome.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_scheduler(optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    """Aquecimento linear seguido de decaimento linear até zero.

    Implementado à mão (e não via ``transformers.get_linear_schedule_with_warmup``)
    porque a localização desse utilitário mudou entre as versões 4.x e 5.x da
    biblioteca — a mesma incompatibilidade que motivou a remoção manual de
    estruturas em ``prune_structured``.
    """
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        restante = total_steps - warmup
        if restante <= 0:
            return 0.0
        return max(0.0, (total_steps - step) / restante)

    return LambdaLR(optimizer, lr_lambda)


def _amp_tools(cfg: Config, enabled: bool):
    """Devolve (autocast_factory, scaler) compatíveis com torch 2.0 e 2.4+.

    A API de precisão mista migrou de ``torch.cuda.amp`` para ``torch.amp``; as
    duas convivem, mas a antiga emite avisos de depreciação nas versões novas.
    """
    usar = enabled and cfg.device == "cuda"
    if not usar:
        return (lambda: nullcontext()), None

    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler()

    def autocast():
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    return autocast, scaler


def finetune(model, blocos: torch.Tensor, cfg: Config, *,
             masks: dict[str, torch.Tensor] | None,
             epochs: float, batch_size: int, grad_accum: int,
             lr: float, weight_decay: float, warmup_ratio: float,
             max_grad_norm: float, max_steps: int | None,
             use_amp: bool, grad_checkpoint: bool,
             eval_every: int, val_input_ids: torch.Tensor | None,
             curva: list[dict] | None, rotulo: str) -> dict:
    """Treina o modelo podado e devolve estatísticas do treino.

    ``masks`` preserva a esparsidade não estruturada; passe ``None`` para poda
    estruturada (ou para o controle denso), onde não há nada a preservar.
    """
    device = cfg.device
    n_blocos = blocos.size(0)
    passos_por_epoca = math.ceil(n_blocos / (batch_size * grad_accum))
    total_steps = int(passos_por_epoca * epochs)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    if total_steps < 1:
        raise ValueError(
            f"orçamento de treino vazio: {n_blocos} blocos, batch {batch_size}, "
            f"acúmulo {grad_accum}, épocas {epochs}"
        )

    # use_cache guarda estados de atenção que só servem para geração e ocupam
    # memória à toa durante o treino.
    cache_original = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    if grad_checkpoint:
        model.gradient_checkpointing_enable()

    optimizer = AdamW(_param_groups(model, weight_decay), lr=lr)
    scheduler = _build_scheduler(optimizer, total_steps, warmup_ratio)
    autocast, scaler = _amp_tools(cfg, use_amp)

    # Gerador próprio para o embaralhamento: torna a ordem dos lotes função
    # apenas da semente, sem depender do estado global do RNG.
    gerador = torch.Generator().manual_seed(cfg.seed)

    print(f"  [treino] {n_blocos} blocos | {passos_por_epoca} passos/época "
          f"| total {total_steps} passos | amp={scaler is not None}")

    model.train()
    perdas: list[float] = []
    step = 0
    inicio = time.perf_counter()
    parar = False

    while not parar:
        ordem = torch.randperm(n_blocos, generator=gerador)
        optimizer.zero_grad(set_to_none=True)

        for micro, comeco in enumerate(range(0, n_blocos, batch_size)):
            lote = blocos[ordem[comeco:comeco + batch_size]].to(device)

            with autocast():
                saida = model(lote, labels=lote)
                # A perda é dividida pelo acúmulo para que o gradiente
                # acumulado seja a MÉDIA dos micro-lotes, não a soma — sem
                # isso o gradiente efetivo sairia ``grad_accum`` vezes maior
                # que o do lote equivalente.
                #
                # Ressalva conhecida: quando o número de blocos não é múltiplo
                # de ``batch_size``, o último micro-lote da época é menor e
                # entra na média com o mesmo peso dos demais, o que o
                # sobre-representa levemente. É a convenção usual e o efeito é
                # de um micro-lote em centenas; corrigir exigiria ponderar
                # pelo número de tokens de cada micro-lote.
                perda = saida.loss / grad_accum

            if scaler is not None:
                scaler.scale(perda).backward()
            else:
                perda.backward()

            perdas.append(float(saida.loss.detach()))

            if (micro + 1) % grad_accum != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # A esparsidade é restaurada aqui, depois do passo — momento e
            # weight decay já agiram, e é isto que desfaz o estrago.
            if masks is not None:
                reapply_masks(model, masks)

            step += 1

            if step % max(1, total_steps // 20) == 0 or step == 1:
                janela = perdas[-50:]
                media = sum(janela) / len(janela)
                print(f"    passo {step:5d}/{total_steps} | perda {media:.4f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")

            if eval_every and step % eval_every == 0 and val_input_ids is not None:
                model.eval()
                ppl_val = compute_perplexity(model, val_input_ids, cfg)
                model.train()
                print(f"    [validação] passo {step}: perplexity {ppl_val:.4f}")
                if curva is not None:
                    curva.append({
                        "experimento": rotulo,
                        "passo": step,
                        "perda_treino": sum(perdas[-50:]) / len(perdas[-50:]),
                        "perplexity_validacao": ppl_val,
                    })

            if step >= total_steps:
                parar = True
                break

    duracao = time.perf_counter() - inicio
    model.eval()
    if grad_checkpoint:
        model.gradient_checkpointing_disable()
    if cache_original is not None:
        model.config.use_cache = cache_original

    return {
        "passos_treino": step,
        "blocos_treino": n_blocos,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "epochs": epochs,
        "tempo_treino_s": duracao,
        "perda_final": sum(perdas[-50:]) / len(perdas[-50:]) if perdas else float("nan"),
    }


# ---------------------------------------------------------------------------
# Experimento completo (uma linha do CSV)
# ---------------------------------------------------------------------------

def run_config(cfg: Config, tokenizer, input_ids, train_blocos, val_input_ids,
               estrategia: str, sparsity: float, *, scope: str, alvo: str,
               beta: float, sens_base: str, curva: list[dict] | None,
               **treino) -> dict:
    """Poda + fine-tuning + avaliação completa de UMA configuração.

    O modelo é recarregado limpo, como nas varreduras one-shot, para que os
    níveis de esparsidade sejam independentes entre si.
    """
    rotulo = (f"finetune_{estrategia}"
              f"_s{int(round(sparsity * 100)):02d}")
    print(f"\n=== {rotulo} ===")

    model, _ = load_model_and_tokenizer(cfg)
    descricao = apply_strategy(cfg, model, estrategia, sparsity, scope,
                               alvo, beta, sens_base)

    # Perplexity one-shot: a linha de partida contra a qual o ganho do
    # fine-tuning é medido. Deve reproduzir a varredura sem fine-tuning.
    ppl_pre = compute_perplexity(model, input_ids, cfg)
    print(f"  perplexity one-shot (antes do treino): {ppl_pre:.4f}")

    # Só a poda não estruturada precisa de máscara; na estruturada as
    # estruturas já não existem, e no controle denso não há o que preservar.
    precisa_mascara = (
        estrategia == "magnitude"
        or (estrategia == "sensibilidade"
            and descricao["escopo"] == "sensibilidade:magnitude")
    )
    masks = capture_masks(model) if precisa_mascara and sparsity > 0 else None

    metrics = {
        "experimento": rotulo,
        "modelo": cfg.model_name,
        "dataset": cfg.dataset_config,
        "estrategia": estrategia,
        "esparsidade_alvo": sparsity,
        "beta": beta if estrategia == "sensibilidade" else "",
        "seed": cfg.seed,
        **descricao,
        "perplexity_pre": ppl_pre,
    }

    # A energia do TREINO é medida à parte da energia da inferência: é um custo
    # pago uma única vez, que precisa ser amortizado contra a economia por
    # inferência para que a conta energética do trabalho feche.
    energia_treino: dict = {}
    with track_energy(cfg, energia_treino):
        estat = finetune(model, train_blocos, cfg, masks=masks,
                         val_input_ids=val_input_ids,
                         curva=curva, rotulo=rotulo, **treino)
    metrics.update(estat)
    metrics["energia_treino_kwh"] = energia_treino.get("energia_kwh", "")
    metrics["emissoes_treino_kg_co2"] = energia_treino.get("emissoes_kg_co2", "")

    # Verificação de integridade da máscara: se a esparsidade não sobreviveu ao
    # treino, o resultado não é o de um modelo podado e a linha é inválida.
    if masks is not None:
        metrics["esparsidade_pos"] = prunable_sparsity(model)
        desvio = abs(metrics["esparsidade_pos"] - descricao["esparsidade_real"])
        if desvio > 1e-6:
            print(f"  [ERRO] esparsidade não preservada: "
                  f"{descricao['esparsidade_real']:.4%} -> "
                  f"{metrics['esparsidade_pos']:.4%}")
        else:
            print(f"  esparsidade preservada: {metrics['esparsidade_pos']:.2%}")
    else:
        metrics["esparsidade_pos"] = descricao["esparsidade_real"]

    # Pré-semeadas para que a posição e a presença das colunas no CSV não
    # dependam de o CodeCarbon estar ativo (o cabeçalho é escrito na primeira
    # linha e vale para todas as seguintes).
    metrics.setdefault("energia_kwh", "")
    metrics.setdefault("emissoes_kg_co2", "")
    with track_energy(cfg, metrics):
        metrics.update(evaluate_all(model, input_ids, cfg))

    # Quanto da degradação foi revertida, em escala log (a mesma do perfil de
    # sensibilidade): positivo = o treino melhorou o modelo.
    metrics["recuperacao_log"] = math.log(ppl_pre / metrics["perplexity"])

    print(f"  perplexity pós-treino: {metrics['perplexity']:.4f} "
          f"(recuperação log {metrics['recuperacao_log']:+.4f})")
    return metrics


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def parse_args():
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Fine-tuning de recuperação pós-poda (GPT-2)."
    )
    p.add_argument("--model-name", default=cfg.model_name)
    p.add_argument("--dataset-config", default=cfg.dataset_config,
                   help="ex.: wikitext-2-raw-v1 ou wikitext-103-raw-v1")
    p.add_argument("--estrategia", default="nenhuma", choices=list(ESTRATEGIAS),
                   help="'nenhuma' = controle denso (obrigatório para interpretar o resto)")
    p.add_argument("--scope", default="uniforme", choices=["global", "uniforme"])
    p.add_argument("--alvo", default="ambos", choices=["cabecas", "mlp", "ambos"],
                   help="alvo da poda estruturada; define também a base do perfil")
    p.add_argument("--beta", type=float, default=1.0,
                   help="expoente da alocação por sensibilidade")
    p.add_argument("--sens-base", default="estruturada",
                   choices=["magnitude", "estruturada"],
                   help="qual perfil guia a alocação (os dois não são intercambiáveis)")
    p.add_argument("--sparsities", type=float, nargs="+",
                   default=[0.3, 0.5],
                   help="níveis a treinar (o controle denso ignora e usa [0.0])")

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="teto de passos do otimizador (útil no limite de sessão do Kaggle)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true",
                   help="desliga a precisão mista (fp16) na GPU")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="troca memória por tempo (útil se faltar VRAM)")

    p.add_argument("--eval-every", type=int, default=0,
                   help="avalia a validação a cada N passos (0 = desligado)")
    p.add_argument("--limit-train-blocks", type=int, default=None,
                   help="usa só os N primeiros blocos de treino (teste de fumaça)")
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no-energy", action="store_true",
                   help="desabilita a medição de energia (CodeCarbon)")
    args = p.parse_args()

    cfg = replace(
        cfg,
        model_name=args.model_name,
        dataset_config=args.dataset_config,
        seed=args.seed,
        energy_enabled=not args.no_energy,
    )
    return cfg, args


def main() -> None:
    cfg, args = parse_args()
    set_seed(cfg.seed)

    print(f"[finetune] dispositivo: {cfg.device}")
    print(f"[finetune] modelo: {cfg.model_name} | dataset: {cfg.dataset_config}")
    print(f"[finetune] estrategia: {args.estrategia} | seed: {cfg.seed}")

    _, tokenizer = load_model_and_tokenizer(cfg)

    input_ids = load_encodings(cfg, tokenizer)
    print(f"[finetune] tokens de avaliação (teste): {input_ids.size(1):,}")

    train_blocos = load_blocks(cfg, tokenizer, "train", args.block_size,
                               args.limit_train_blocks)
    print(f"[finetune] blocos de treino: {train_blocos.size(0):,} "
          f"x {args.block_size} tokens")

    val_input_ids = None
    if args.eval_every:
        val_input_ids = load_encodings(replace(cfg, dataset_split="validation"),
                                       tokenizer)
        print(f"[finetune] tokens de validação: {val_input_ids.size(1):,}")

    # O controle denso é uma configuração única: não há esparsidade a varrer.
    sparsities = [0.0] if args.estrategia == "nenhuma" else args.sparsities

    treino = dict(
        epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm, max_steps=args.max_steps,
        use_amp=not args.no_amp, grad_checkpoint=args.grad_checkpoint,
        eval_every=args.eval_every,
    )

    curva: list[dict] = []
    path = None
    for sparsity in sparsities:
        # Cada configuração parte da mesma semente: as diferenças entre linhas
        # vêm da poda, não da ordem dos lotes.
        set_seed(cfg.seed)
        metrics = run_config(cfg, tokenizer, input_ids, train_blocos,
                             val_input_ids, args.estrategia, sparsity,
                             scope=args.scope, alvo=args.alvo, beta=args.beta,
                             sens_base=args.sens_base, curva=curva, **treino)
        path = append_result(cfg.results_dir, RESULTADOS_CSV, metrics)

    for linha in curva:
        append_result(cfg.results_dir, CURVA_CSV, linha)

    if path is None:
        print("\n[finetune] nada a treinar")
    else:
        print(f"\n[finetune] resultados anexados em: {path}")
        if curva:
            print(f"[finetune] curva de recuperação em: "
                  f"{cfg.results_dir}/{CURVA_CSV}")


if __name__ == "__main__":
    main()
