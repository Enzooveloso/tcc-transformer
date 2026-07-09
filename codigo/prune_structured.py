"""Poda estruturada — cabeças de atenção e neurônios MLP (Li et al., 2017).

Segunda das duas estratégias clássicas da metodologia (granularidade grossa).
Em vez de zerar pesos individuais, remove-se estruturas inteiras: cabeças de
atenção e/ou neurônios da camada intermediária do MLP. O critério de
importância é a norma L1 dos pesos de cada estrutura — a adaptação natural,
para o Transformer, do critério que Li et al. (2017) aplicaram a filtros
convolucionais: estruturas cujos pesos têm menor magnitude agregada tendem a
contribuir menos para a saída.

A diferença decisiva em relação à poda por magnitude (``prune_magnitude.py``)
é que aqui as estruturas são **fisicamente removidas** das matrizes — as
projeções ficam menores, e não densas com zeros. Por isso esta é a estratégia
que pode converter redução de parâmetros em redução *real* de FLOPs, tempo e
energia na GPU (pergunta empírica (A) do trabalho): o total de parâmetros, o
tamanho em MB e os FLOPs medidos devem cair junto com a esparsidade, ao
contrário do observado na poda não estruturada.

Uso como biblioteca (gancho do ``run.py``):
    from prune_structured import prune_structured
    model = prune_structured(model, sparsity=0.3)                    # ambos, global
    model = prune_structured(model, sparsity=0.3, alvo="cabecas",
                             scope="uniforme")

Uso como script (varredura completa perplexity x esparsidade):
    python prune_structured.py
    python prune_structured.py --alvo mlp --scope uniforme --sparsities 0 0.3 0.5
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from config import Config
from data import load_encodings
from energy import track_energy
from eval import evaluate_all
from model import load_model_and_tokenizer
from utils import append_result, set_seed

# O GPT-2 implementa as projeções como ``Conv1D`` (linear com pesos
# transpostos). A remoção física é implementada aqui mesmo — a API de poda do
# transformers (``prune_heads``/``prune_conv1d_layer``) foi removida na v5,
# então não dá para depender dela entre versões (local × Kaggle).
try:  # local do símbolo varia entre versões do transformers
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    from transformers.modeling_utils import Conv1D  # type: ignore

ALVOS = ("cabecas", "mlp", "ambos")


# ---------------------------------------------------------------------------
# Importância das estruturas (norma L1, Li et al., 2017)
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_importances(model) -> torch.Tensor:
    """Norma L1 dos pesos de cada cabeça de atenção; forma (n_layers, n_heads).

    Cada cabeça é dona de uma fatia de ``head_dim`` colunas em cada uma das
    projeções Q, K e V do ``c_attn`` e das ``head_dim`` linhas correspondentes
    do ``c_proj``. A importância é a soma dos valores absolutos de todos esses
    pesos (e vieses).
    """
    cfg = model.config
    n_embd, n_heads = cfg.n_embd, cfg.n_head
    head_dim = n_embd // n_heads

    scores = []
    for block in model.transformer.h:
        attn = block.attn
        per_head = torch.zeros(n_heads, device=attn.c_attn.weight.device)

        # c_attn empilha Q, K e V lado a lado: peso (n_embd, 3*n_embd).
        w_qkv = attn.c_attn.weight.split(n_embd, dim=1)
        b_qkv = attn.c_attn.bias.split(n_embd, dim=0)
        for w, b in zip(w_qkv, b_qkv):
            per_head += w.abs().reshape(n_embd, n_heads, head_dim).sum(dim=(0, 2))
            per_head += b.abs().reshape(n_heads, head_dim).sum(dim=1)

        # c_proj recebe as cabeças concatenadas: cada uma ocupa head_dim linhas.
        per_head += (
            attn.c_proj.weight.abs().reshape(n_heads, head_dim, n_embd).sum(dim=(1, 2))
        )
        scores.append(per_head)

    return torch.stack(scores)


@torch.no_grad()
def mlp_importances(model) -> torch.Tensor:
    """Norma L1 dos pesos de cada neurônio MLP; forma (n_layers, n_inner).

    O neurônio ``j`` da camada intermediária é definido pela coluna ``j`` de
    ``c_fc`` (que o produz) e pela linha ``j`` de ``c_proj`` (que o consome).
    """
    scores = []
    for block in model.transformer.h:
        mlp = block.mlp
        s = mlp.c_fc.weight.abs().sum(dim=0)
        s = s + mlp.c_fc.bias.abs()
        s = s + mlp.c_proj.weight.abs().sum(dim=1)
        scores.append(s)
    return torch.stack(scores)


# ---------------------------------------------------------------------------
# Seleção e remoção das estruturas
# ---------------------------------------------------------------------------

def select_structures(scores: torch.Tensor, sparsity: float,
                      scope: str) -> list[list[int]]:
    """Escolhe, por camada, os índices das estruturas a remover.

    ``scores`` tem forma (n_layers, n_structs). Como na poda por magnitude,
    ``scope`` define onde o corte é decidido:
      - ``"global"``   — as ``sparsity``% estruturas de menor norma no modelo
                         inteiro; camadas com estruturas fracas cedem mais.
      - ``"uniforme"`` — o mesmo percentual em cada camada, isoladamente.

    Guarda de segurança: nenhuma camada fica com menos de uma estrutura —
    remover todas as cabeças (ou todos os neurônios) de um bloco quebraria o
    fluxo residual do modelo.
    """
    n_layers, n_structs = scores.shape
    to_remove: list[list[int]] = [[] for _ in range(n_layers)]
    if sparsity <= 0:
        return to_remove

    if scope == "uniforme":
        k = min(int(sparsity * n_structs), n_structs - 1)
        for layer in range(n_layers):
            order = torch.argsort(scores[layer])
            to_remove[layer] = order[:k].tolist()
    elif scope == "global":
        k = int(sparsity * scores.numel())
        order = torch.argsort(scores.flatten())
        removed = 0
        for pos in order.tolist():
            if removed >= k:
                break
            layer, idx = divmod(pos, n_structs)
            if len(to_remove[layer]) >= n_structs - 1:
                continue  # guarda: pula estruturas de camadas já no limite
            to_remove[layer].append(idx)
            removed += 1
    else:
        raise ValueError(f"scope inválido: {scope!r} (use 'global' ou 'uniforme')")

    return to_remove


@torch.no_grad()
def _prune_conv1d(layer: Conv1D, keep: torch.Tensor, dim: int) -> Conv1D:
    """Reconstrói uma ``Conv1D`` mantendo apenas os índices ``keep`` em ``dim``.

    O peso da ``Conv1D`` tem forma (entrada, saída): ``dim=1`` encolhe a saída
    (o viés acompanha), ``dim=0`` encolhe a entrada (viés intacto). Os
    parâmetros sobreviventes continuam treináveis (gancho do fine-tuning).
    """
    keep = keep.to(layer.weight.device)
    weight = layer.weight.index_select(dim, keep).clone().detach()
    bias = (layer.bias.index_select(0, keep) if dim == 1 else layer.bias)
    bias = bias.clone().detach()

    new = Conv1D(weight.size(1), weight.size(0))
    new.weight = torch.nn.Parameter(weight)
    new.bias = torch.nn.Parameter(bias)
    return new


@torch.no_grad()
def remove_heads(model, heads_per_layer: list[list[int]]) -> int:
    """Remove fisicamente as cabeças indicadas; devolve quantas foram removidas.

    Em cada camada, mantém apenas as fatias de ``head_dim`` colunas das cabeças
    sobreviventes nas projeções Q, K e V do ``c_attn`` (e as linhas
    correspondentes do ``c_proj``), e atualiza a contabilidade interna da
    atenção (``num_heads``, ``split_size``) — o forward do GPT-2 deduz o número
    de cabeças das formas dos tensores.
    """
    removed = 0
    for layer, idxs in enumerate(heads_per_layer):
        if not idxs:
            continue
        attn = model.transformer.h[layer].attn
        head_dim = attn.head_dim
        kept = sorted(set(range(attn.num_heads)) - set(idxs))

        # Dimensões internas (colunas de Q, K ou V) das cabeças mantidas.
        dims = torch.cat(
            [torch.arange(h * head_dim, (h + 1) * head_dim) for h in kept]
        )
        # No c_attn, Q, K e V ficam lado a lado, cada um com split_size colunas.
        qkv = torch.cat([dims, dims + attn.split_size, dims + 2 * attn.split_size])

        attn.c_attn = _prune_conv1d(attn.c_attn, qkv, dim=1)
        attn.c_proj = _prune_conv1d(attn.c_proj, dims, dim=0)
        attn.num_heads = len(kept)
        attn.split_size = len(kept) * head_dim
        removed += len(idxs)
    return removed


@torch.no_grad()
def remove_mlp_neurons(model, neurons_per_layer: list[list[int]]) -> int:
    """Remove fisicamente os neurônios MLP indicados; devolve quantos removeu.

    Mantém em cada camada apenas as colunas de ``c_fc`` (e linhas de
    ``c_proj``) dos neurônios sobreviventes — as matrizes encolhem de fato.
    """
    removed = 0
    for layer, idxs in enumerate(neurons_per_layer):
        if not idxs:
            continue
        mlp = model.transformer.h[layer].mlp
        n_inner = mlp.c_fc.weight.size(1)
        keep = torch.tensor(
            sorted(set(range(n_inner)) - set(idxs)), dtype=torch.long
        )
        mlp.c_fc = _prune_conv1d(mlp.c_fc, keep, dim=1)
        mlp.c_proj = _prune_conv1d(mlp.c_proj, keep, dim=0)
        removed += len(idxs)
    return removed


@torch.no_grad()
def prune_structured(model, sparsity: float, scope: str = "global",
                     alvo: str = "ambos"):
    """Aplica poda estruturada, in-place, e devolve o próprio modelo.

    ``sparsity`` é a fração de estruturas a remover (0.0–1.0), aplicada a cada
    alvo separadamente: 0.3 com ``alvo="ambos"`` remove ~30% das cabeças E
    ~30% dos neurônios MLP.

    ``alvo`` escolhe a granularidade estrutural:
      - ``"cabecas"`` — apenas cabeças de atenção;
      - ``"mlp"``     — apenas neurônios da camada intermediária do MLP;
      - ``"ambos"``   — os dois, com a mesma fração (padrão).
    """
    if alvo not in ALVOS:
        raise ValueError(f"alvo inválido: {alvo!r} (use {', '.join(ALVOS)})")

    # As importâncias são calculadas no modelo intacto, antes de qualquer
    # remoção; cabeças e neurônios vivem em módulos independentes, então a
    # ordem de remoção não interfere.
    if alvo in ("cabecas", "ambos"):
        heads = select_structures(head_importances(model), sparsity, scope)
        remove_heads(model, heads)
    if alvo in ("mlp", "ambos"):
        neurons = select_structures(mlp_importances(model), sparsity, scope)
        remove_mlp_neurons(model, neurons)
    return model


def prunable_param_count(model) -> int:
    """Total de parâmetros das projeções sujeitas à poda estruturada.

    Comparar esse total antes e depois da poda dá a esparsidade *real* em
    parâmetros — aqui os parâmetros somem de verdade, em vez de virarem zeros.
    """
    total = 0
    for block in model.transformer.h:
        for module in (block.attn.c_attn, block.attn.c_proj,
                       block.mlp.c_fc, block.mlp.c_proj):
            total += sum(p.numel() for p in module.parameters())
    return total


# ---------------------------------------------------------------------------
# Varredura (script)
# ---------------------------------------------------------------------------

def parse_args() -> tuple[Config, str, str, list[float]]:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Varredura de poda estruturada (GPT-2)."
    )
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--dataset-config", default=cfg.dataset_config,
                        help="ex.: wikitext-2-raw-v1 ou wikitext-103-raw-v1")
    parser.add_argument("--scope", default="global", choices=["global", "uniforme"],
                        help="global (corte único) ou uniforme (mesmo %% por camada)")
    parser.add_argument("--alvo", default="ambos", choices=list(ALVOS),
                        help="estruturas a remover: cabecas, mlp ou ambos")
    parser.add_argument("--sparsities", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="frações de estruturas a remover, por alvo")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--no-energy", action="store_true",
                        help="desabilita a medição de energia (CodeCarbon)")
    args = parser.parse_args()

    cfg = replace(
        cfg,
        model_name=args.model_name,
        dataset_config=args.dataset_config,
        seed=args.seed,
        energy_enabled=not args.no_energy,
    )
    return cfg, args.scope, args.alvo, args.sparsities


def evaluate_at_sparsity(cfg: Config, input_ids, sparsity: float, scope: str,
                         alvo: str) -> dict:
    """Carrega um modelo limpo, poda-o à ``sparsity`` alvo e coleta as métricas.

    Como na magnitude, o modelo é recarregado do zero a cada nível para que os
    experimentos sejam independentes (sem poda cumulativa).
    """
    model, _ = load_model_and_tokenizer(cfg)
    params_before = prunable_param_count(model)

    heads_removed = 0
    neurons_removed = 0
    if alvo in ("cabecas", "ambos"):
        heads = select_structures(head_importances(model), sparsity, scope)
        heads_removed = remove_heads(model, heads)
    if alvo in ("mlp", "ambos"):
        neurons = select_structures(mlp_importances(model), sparsity, scope)
        neurons_removed = remove_mlp_neurons(model, neurons)

    params_after = prunable_param_count(model)

    metrics = {
        "experimento": f"estruturada_{alvo}_{scope}_s{int(round(sparsity * 100)):02d}",
        "modelo": cfg.model_name,
        "dataset": cfg.dataset_config,
        "escopo": scope,
        "alvo": alvo,
        "esparsidade_alvo": sparsity,
        "esparsidade_real": 1.0 - params_after / params_before,
        "cabecas_removidas": heads_removed,
        "neuronios_removidos": neurons_removed,
    }
    with track_energy(cfg, metrics):
        metrics.update(evaluate_all(model, input_ids, cfg))
    return metrics


def main() -> None:
    cfg, scope, alvo, sparsities = parse_args()
    set_seed(cfg.seed)

    print(f"[estruturada] dispositivo: {cfg.device}")
    print(f"[estruturada] modelo: {cfg.model_name} | dataset: {cfg.dataset_config}")
    print(f"[estruturada] escopo: {scope} | alvo: {alvo} | esparsidades: {sparsities}")

    # Tokeniza o conjunto de avaliação uma única vez (não depende da poda).
    _, tokenizer = load_model_and_tokenizer(cfg)
    input_ids = load_encodings(cfg, tokenizer)
    print(f"[estruturada] tokens de avaliação: {input_ids.size(1):,}")

    for sparsity in sparsities:
        metrics = evaluate_at_sparsity(cfg, input_ids, sparsity, scope, alvo)
        print(f"\n--- esparsidade alvo {sparsity:.0%} "
              f"(real em params {metrics['esparsidade_real']:.2%}) ---")
        print(f"  perplexity: {metrics['perplexity']:.4f} | "
              f"cabecas: -{metrics['cabecas_removidas']} | "
              f"neuronios: -{metrics['neuronios_removidos']} | "
              f"flops: {metrics['flops']:.3e}")
        path = append_result(cfg.results_dir, "estruturada.csv", metrics)

    print(f"\n[estruturada] varredura anexada em: {path}")


if __name__ == "__main__":
    main()
