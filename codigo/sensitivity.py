"""Análise de sensibilidade por camada — a estratégia de OTIMIZAÇÃO do trabalho.

Os experimentos com as duas estratégias clássicas mostraram que nenhuma
alocação ingênua de esparsidade vence sempre: na poda por magnitude o escopo
uniforme dominou a faixa útil, na estruturada foi o global — e ambos derretem
em algum regime. A hipótese central do TCC é que a causa é a mesma: as camadas
do Transformer NÃO são igualmente importantes, e tratar todas igual (uniforme)
ou confiar cegamente na magnitude/norma (global) aloca mal o orçamento de poda.

Este módulo testa a hipótese em duas fases:

1. **Perfil** — para cada camada, poda-se APENAS ela a uma taxa-sonda fixa
   (mantendo o resto intacto) e mede-se a degradação de perplexity. O
   resultado é o perfil de sensibilidade do GPT-2, camada a camada, para cada
   estratégia (magnitude e estruturada). Grava em ``sensibilidade_perfil.csv``.

2. **Varredura** — o perfil vira uma alocação de esparsidade: camadas robustas
   (pouca degradação) recebem taxas maiores, camadas sensíveis recebem taxas
   menores, com a média ponderada por parâmetros respeitando o orçamento
   global. A alocação é avaliada com o vetor completo de métricas, nos mesmos
   níveis das varreduras ingênuas, e grava em ``sensibilidade.csv`` — o
   terceiro competidor das curvas do capítulo de Resultados.

Uso como script:
    python sensitivity.py                        # perfil + varredura, 2 estratégias
    python sensitivity.py --fase perfil          # só o perfil
    python sensitivity.py --fase varredura       # só a varredura (lê o perfil do CSV)
    python sensitivity.py --estrategias magnitude --sparsities 0 0.3 0.5
    python sensitivity.py --fase varredura --beta 0.5   # ablação da alocação

O quanto o perfil é levado a sério na fase 2 é controlado por ``--beta``: 0
recai na poda uniforme, 1 (padrão) usa o inverso puro da degradação medida.
Rodar a varredura com dois betas, sobre o MESMO perfil, dá a ablação da função
de alocação sem refazer a fase 1 (que é a cara).
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace

import torch

from config import Config
from data import load_encodings
from energy import track_energy
from eval import compute_perplexity, evaluate_all
from model import load_model_and_tokenizer
from prune_magnitude import _magnitude_threshold, prunable_sparsity
from prune_structured import (
    head_importances,
    mlp_importances,
    prunable_param_count,
    remove_heads,
    remove_mlp_neurons,
)
from utils import append_result, set_seed

ESTRATEGIAS = ("magnitude", "estruturada")

PERFIL_CSV = "sensibilidade_perfil.csv"
VARREDURA_CSV = "sensibilidade.csv"

# Teto de taxa por camada: mesmo a camada mais robusta nunca é (quase) zerada.
TAXA_MAX = 0.95

# Expoente da alocação: taxa ~ robustez^BETA. Com 1.0 a taxa é o inverso puro
# do delta de log-perplexity; com 0.0 a alocação degenera na poda uniforme.
# Valores intermediários amortecem a razão entre camadas (ver ``allocate_rates``).
BETA = 1.0


# ---------------------------------------------------------------------------
# Poda com taxa individual por camada (as duas estratégias)
# ---------------------------------------------------------------------------

def _block_weights(block) -> list[torch.Tensor]:
    """Matrizes podáveis de um bloco: as mesmas 4 projeções das varreduras."""
    return [
        block.attn.c_attn.weight,
        block.attn.c_proj.weight,
        block.mlp.c_fc.weight,
        block.mlp.c_proj.weight,
    ]


@torch.no_grad()
def prune_magnitude_per_layer(model, rates: list[float]):
    """Poda por magnitude com uma taxa própria para cada camada.

    Dentro de cada bloco, o limiar é único para as 4 projeções (o análogo do
    escopo global, restrito à camada); entre blocos, cada um segue sua taxa.
    """
    for block, rate in zip(model.transformer.h, rates):
        if rate <= 0:
            continue
        weights = _block_weights(block)
        scores = torch.cat([w.abs().flatten() for w in weights])
        threshold = _magnitude_threshold(scores, rate)
        for w in weights:
            w.mul_((w.abs() > threshold).to(w.dtype))
    return model


@torch.no_grad()
def prune_structured_per_layer(model, rates: list[float]) -> tuple[int, int]:
    """Poda estruturada (cabeças + neurônios MLP) com taxa própria por camada.

    A mesma taxa da camada vale para os dois alvos, como na varredura com
    ``alvo="ambos"``. Devolve (cabeças removidas, neurônios removidos).
    """
    h_scores = head_importances(model)
    n_scores = mlp_importances(model)
    n_heads, n_neurons = h_scores.size(1), n_scores.size(1)

    heads, neurons = [], []
    for layer, rate in enumerate(rates):
        k_h = min(int(rate * n_heads), n_heads - 1)
        k_n = min(int(rate * n_neurons), n_neurons - 1)
        heads.append(torch.argsort(h_scores[layer])[:k_h].tolist())
        neurons.append(torch.argsort(n_scores[layer])[:k_n].tolist())

    removed_h = remove_heads(model, heads)
    removed_n = remove_mlp_neurons(model, neurons)
    return removed_h, removed_n


def apply_rates(model, estrategia: str, rates: list[float]) -> tuple[int, int]:
    """Aplica a estratégia com as taxas por camada; devolve contagens removidas."""
    if estrategia == "magnitude":
        prune_magnitude_per_layer(model, rates)
        return 0, 0
    if estrategia == "estruturada":
        return prune_structured_per_layer(model, rates)
    raise ValueError(f"estrategia inválida: {estrategia!r} (use {ESTRATEGIAS})")


# ---------------------------------------------------------------------------
# Fase 1 — perfil de sensibilidade
# ---------------------------------------------------------------------------

def profile_layer(cfg: Config, input_ids, estrategia: str, layer: int,
                  taxa: float, ppl_base: float) -> dict:
    """Poda apenas ``layer`` à taxa-sonda e mede a perplexity resultante.

    Só a perplexity interessa aqui (o perfil é diagnóstico, não uma linha de
    resultados) — sem medição de tempo/energia, o perfil fica ~3x mais barato.
    """
    model, _ = load_model_and_tokenizer(cfg)
    n_layers = len(model.transformer.h)
    rates = [taxa if i == layer else 0.0 for i in range(n_layers)]
    apply_rates(model, estrategia, rates)

    ppl = compute_perplexity(model, input_ids, cfg)
    return {
        "experimento": f"perfil_{estrategia}_c{layer:02d}",
        "modelo": cfg.model_name,
        "dataset": cfg.dataset_config,
        "estrategia": estrategia,
        "camada": layer,
        "taxa_sonda": taxa,
        "perplexity": ppl,
        "perplexity_baseline": ppl_base,
        # Log para domar as explosões de perplexity; é o score da alocação.
        "delta_log_ppl": float(torch.log(torch.tensor(ppl / ppl_base))),
    }


def run_profile(cfg: Config, input_ids, estrategias: list[str],
                taxa: float) -> None:
    model, _ = load_model_and_tokenizer(cfg)
    n_layers = len(model.transformer.h)
    ppl_base = compute_perplexity(model, input_ids, cfg)
    del model
    print(f"[sensibilidade] perplexity baseline: {ppl_base:.4f}")

    for estrategia in estrategias:
        print(f"\n[sensibilidade] perfil — {estrategia} (taxa-sonda {taxa:.0%})")
        for layer in range(n_layers):
            row = profile_layer(cfg, input_ids, estrategia, layer, taxa, ppl_base)
            print(f"  camada {layer:2d}: perplexity {row['perplexity']:10.2f} "
                  f"(delta log {row['delta_log_ppl']:+.4f})")
            append_result(cfg.results_dir, PERFIL_CSV, row)


def load_profile(cfg: Config, estrategia: str) -> tuple[list[float], str]:
    """Lê os scores de sensibilidade (delta_log_ppl) por camada do perfil.

    Só entram as linhas do mesmo modelo E do mesmo dataset da ``cfg`` — o
    ``dataset_config`` é plugável (WikiText-2 / WikiText-103) e perfis de
    corpora diferentes não podem ser promediados.

    Havendo mais de uma taxa-sonda por camada, os deltas são promediados (o que
    importa para a alocação é a ordem relativa entre camadas), mas isso é
    avisado no log: normalmente indica um perfil rerodado com outra sonda sobre
    o CSV antigo, e não uma decisão deliberada.

    Devolve os deltas por camada e o rótulo das taxas-sonda que os geraram.
    """
    path = os.path.join(cfg.results_dir, PERFIL_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} não existe — rode antes: python sensitivity.py --fase perfil"
        )
    somas: dict[int, float] = {}
    contagens: dict[int, int] = {}
    sondas: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["estrategia"] != estrategia
                    or row["modelo"] != cfg.model_name
                    or row["dataset"] != cfg.dataset_config):
                continue
            layer = int(row["camada"])
            somas[layer] = somas.get(layer, 0.0) + float(row["delta_log_ppl"])
            contagens[layer] = contagens.get(layer, 0) + 1
            sondas.add(row["taxa_sonda"])
    if not somas:
        raise ValueError(
            f"perfil sem linhas para a estratégia {estrategia!r} "
            f"em {cfg.model_name}/{cfg.dataset_config}"
        )
    layers = sorted(somas)
    if layers != list(range(len(layers))):
        raise ValueError(f"perfil incompleto: camadas presentes = {layers}")

    rotulo = ";".join(sorted(sondas, key=float))
    if len(sondas) > 1:
        print(f"  [aviso] o perfil de {estrategia} mistura {len(sondas)} "
              f"taxas-sonda ({rotulo}); os deltas serão promediados. "
              f"Se não foi intencional, limpe o {PERFIL_CSV} e refaça o perfil.")
    return [somas[l] / contagens[l] for l in layers], rotulo


# ---------------------------------------------------------------------------
# Fase 2 — alocação de esparsidade guiada pelo perfil
# ---------------------------------------------------------------------------

def allocate_rates(deltas: list[float], weights: list[float], target: float,
                   beta: float = BETA, rate_max: float = TAXA_MAX,
                   eps: float = 1e-4) -> list[float]:
    """Converte o perfil de sensibilidade em taxas de poda por camada.

    A taxa de cada camada é proporcional à sua robustez elevada a ``beta`` — a
    robustez sendo o inverso do delta de log-perplexity medido pela sonda —,
    escalada por um fator único ``alpha`` tal que a média das taxas, ponderada
    pelos parâmetros de cada camada, atinja o orçamento ``target``. Como as
    taxas são limitadas a ``rate_max``, ``alpha`` é encontrado por bisseção (a
    fração podada é monótona em ``alpha``).

    ``beta`` controla o quanto o perfil é levado a sério: com ``0.0`` a
    alocação vira a poda uniforme (todas as camadas com a mesma taxa), com
    ``1.0`` é o inverso puro do delta, e valores intermediários amortecem a
    razão entre camadas. O amortecimento importa porque a razão de robustez é
    ilimitada: uma camada com delta 0.001 receberia taxa 1000x maior que uma
    com delta 1.0, concentrando o orçamento inteiro em uma ou duas camadas.

    Deltas negativos (a poda da sonda *melhorou* a perplexity — acontece com
    cabeças de atenção, cf. Michel et al., 2019) são tratados como robustez
    máxima pelo piso ``eps``.
    """
    if target <= 0:
        return [0.0] * len(deltas)

    robustez = [(1.0 / max(d, eps)) ** beta for d in deltas]
    total = sum(weights)

    def fracao_podada(alpha: float) -> float:
        return sum(w * min(alpha * r, rate_max)
                   for w, r in zip(weights, robustez)) / total

    hi = 1.0
    while fracao_podada(hi) < target and hi < 1e12:
        hi *= 2.0
    if fracao_podada(hi) < target:  # orçamento acima do teto: satura tudo
        return [rate_max] * len(deltas)

    lo = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if fracao_podada(mid) < target:
            lo = mid
        else:
            hi = mid
    return [min(hi * r, rate_max) for r in robustez]


def _layer_param_weights(model) -> list[float]:
    """Parâmetros podáveis por camada (pesos da média ponderada da alocação)."""
    return [float(sum(w.numel() for w in _block_weights(block)))
            for block in model.transformer.h]


def evaluate_allocation(cfg: Config, input_ids, estrategia: str, target: float,
                        rates: list[float], taxa_sonda: str, beta: float) -> dict:
    """Modelo limpo + poda com as taxas alocadas + vetor completo de métricas."""
    model, _ = load_model_and_tokenizer(cfg)
    params_before = prunable_param_count(model)

    heads_removed, neurons_removed = apply_rates(model, estrategia, rates)

    if estrategia == "magnitude":
        esparsidade_real = prunable_sparsity(model)
    else:
        esparsidade_real = 1.0 - prunable_param_count(model) / params_before

    metrics = {
        "experimento": f"sensibilidade_{estrategia}_s{int(round(target * 100)):02d}",
        "modelo": cfg.model_name,
        "dataset": cfg.dataset_config,
        "estrategia": estrategia,
        "escopo": "sensibilidade",
        "esparsidade_alvo": target,
        "esparsidade_real": esparsidade_real,
        "cabecas_removidas": heads_removed,
        "neuronios_removidos": neurons_removed,
        # Hiperparâmetros que geraram esta alocação: sem eles a linha não é
        # reproduzível (o perfil e o expoente definem as taxas por camada).
        "taxa_sonda": taxa_sonda,
        "beta": beta,
        "taxas_por_camada": ";".join(f"{r:.4f}" for r in rates),
    }
    with track_energy(cfg, metrics):
        metrics.update(evaluate_all(model, input_ids, cfg))
    return metrics


def run_sweep(cfg: Config, input_ids, estrategias: list[str],
              sparsities: list[float], beta: float = BETA,
              eps: float = 1e-4) -> None:
    model, _ = load_model_and_tokenizer(cfg)
    weights = _layer_param_weights(model)
    n_layers = len(weights)
    del model

    path = None
    for estrategia in estrategias:
        deltas, taxa_sonda = load_profile(cfg, estrategia)
        if len(deltas) != n_layers:
            raise ValueError(
                f"perfil de {estrategia} tem {len(deltas)} camadas; modelo tem {n_layers}"
            )
        print(f"\n[sensibilidade] varredura — {estrategia} (beta {beta})")
        print(f"  deltas do perfil: {['%.3f' % d for d in deltas]}")

        # Camadas cujo delta caiu no piso: a sonda não as degradou (ou até as
        # melhorou), então entram na alocação com robustez máxima. Muitas
        # camadas no piso = perfil pouco informativo, e a alocação tende a
        # concentrar o orçamento nelas.
        no_piso = [i for i, d in enumerate(deltas) if d <= eps]
        if no_piso:
            print(f"  [aviso] {len(no_piso)}/{n_layers} camadas com delta <= {eps} "
                  f"(robustez máxima, saturam primeiro): {no_piso}")

        for target in sparsities:
            rates = allocate_rates(deltas, weights, target, beta=beta, eps=eps)
            metrics = evaluate_allocation(cfg, input_ids, estrategia, target,
                                          rates, taxa_sonda, beta)
            saturadas = [i for i, r in enumerate(rates) if r >= TAXA_MAX - 1e-9]
            print(f"\n--- {estrategia} | orçamento {target:.0%} "
                  f"(real {metrics['esparsidade_real']:.2%}) ---")
            print(f"  taxas: {['%.2f' % r for r in rates]}")
            # Alocação degenerada (poucas camadas no teto absorvendo o
            # orçamento) fica visível aqui, no log, e não só depois no CSV.
            print(f"  camadas no teto ({TAXA_MAX:.0%}): {len(saturadas)}/{n_layers}"
                  + (f" -> {saturadas}" if saturadas else ""))
            print(f"  perplexity: {metrics['perplexity']:.4f}")
            path = append_result(cfg.results_dir, VARREDURA_CSV, metrics)

    if path is None:
        print("\n[sensibilidade] nada a varrer (sem estratégias ou sem orçamentos)")
    else:
        print(f"\n[sensibilidade] varredura anexada em: {path}")


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def parse_args() -> tuple[Config, str, list[str], float, list[float], float]:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Análise de sensibilidade por camada (GPT-2)."
    )
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--dataset-config", default=cfg.dataset_config,
                        help="ex.: wikitext-2-raw-v1 ou wikitext-103-raw-v1")
    parser.add_argument("--fase", default="completa",
                        choices=["perfil", "varredura", "completa"],
                        help="perfil (fase 1), varredura (fase 2) ou completa")
    parser.add_argument("--estrategias", nargs="+", default=list(ESTRATEGIAS),
                        choices=list(ESTRATEGIAS))
    parser.add_argument("--taxa-sonda", type=float, default=0.5,
                        help="taxa aplicada a cada camada isolada no perfil")
    parser.add_argument("--beta", type=float, default=BETA,
                        help="expoente da alocação (0 = uniforme, 1 = inverso "
                             "puro do delta); amortece a razão entre camadas")
    parser.add_argument("--sparsities", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="orçamentos globais de esparsidade da varredura")
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
    return (cfg, args.fase, args.estrategias, args.taxa_sonda, args.sparsities,
            args.beta)


def main() -> None:
    cfg, fase, estrategias, taxa, sparsities, beta = parse_args()
    set_seed(cfg.seed)

    print(f"[sensibilidade] dispositivo: {cfg.device}")
    print(f"[sensibilidade] modelo: {cfg.model_name} | dataset: {cfg.dataset_config}")
    print(f"[sensibilidade] fase: {fase} | estrategias: {estrategias} | beta: {beta}")

    _, tokenizer = load_model_and_tokenizer(cfg)
    input_ids = load_encodings(cfg, tokenizer)
    print(f"[sensibilidade] tokens de avaliação: {input_ids.size(1):,}")

    if fase in ("perfil", "completa"):
        run_profile(cfg, input_ids, estrategias, taxa)
    if fase in ("varredura", "completa"):
        run_sweep(cfg, input_ids, estrategias, sparsities, beta=beta)


if __name__ == "__main__":
    main()
