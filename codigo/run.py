"""Orquestrador da pipeline de poda neural — GPT-2 small.

Estágio 0 (implementado): avalia o modelo de referência (baseline) e registra
o vetor completo de métricas — perplexity, custo computacional e energia —
como a primeira linha da tabela de resultados.

Estágios seguintes (ver ganchos ao final) reaproveitam exatamente este fluxo,
apenas inserindo o passo de poda entre o carregamento e a avaliação.

Exemplos:
    python run.py                      # baseline em WikiText-2 (padrão)
    python run.py --dataset-config wikitext-103-raw-v1
    python run.py --no-energy          # desliga o CodeCarbon
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from config import Config
from data import load_encodings
from eval import evaluate_all
from energy import track_energy
from model import load_model_and_tokenizer
from utils import append_result, set_seed


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Pipeline de poda neural (GPT-2).")
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--dataset-config", default=cfg.dataset_config,
                        help="ex.: wikitext-2-raw-v1 ou wikitext-103-raw-v1")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--no-energy", action="store_true",
                        help="desabilita a medição de energia (CodeCarbon)")
    args = parser.parse_args()

    return replace(
        cfg,
        model_name=args.model_name,
        dataset_config=args.dataset_config,
        seed=args.seed,
        energy_enabled=not args.no_energy,
    )


def run_baseline(cfg: Config) -> dict:
    """Carrega o baseline, avalia e devolve o dicionário de métricas."""
    set_seed(cfg.seed)

    print(f"[run] dispositivo: {cfg.device}")
    print(f"[run] modelo: {cfg.model_name} | dataset: {cfg.dataset_config}")

    model, tokenizer = load_model_and_tokenizer(cfg)
    input_ids = load_encodings(cfg, tokenizer)
    print(f"[run] tokens de avaliação: {input_ids.size(1):,}")

    metrics = {"experimento": "baseline", "modelo": cfg.model_name,
               "dataset": cfg.dataset_config}

    # Envolve a avaliação na medição de energia (bloco representativo).
    with track_energy(cfg, metrics):
        metrics.update(evaluate_all(model, input_ids, cfg))

    return metrics


def main() -> None:
    cfg = parse_args()
    metrics = run_baseline(cfg)

    print("\n=== Métricas do baseline ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    path = append_result(cfg.results_dir, "resultados.csv", metrics)
    print(f"\n[run] métricas anexadas em: {path}")

    # ------------------------------------------------------------------
    # GANCHOS PARA OS PRÓXIMOS ESTÁGIOS (ainda não implementados):
    #
    #   Estágio 1 — poda sem otimização:
    #     from prune_magnitude import prune_magnitude
    #     model = prune_magnitude(model, sparsity=0.5)
    #     # (opcional) fine-tuning one-shot -> depois:
    #     # from finetune import finetune; model = finetune(model, ...)
    #     metrics = evaluate_all(model, input_ids, cfg)
    #
    #   Estágio 2 — poda com otimização:
    #     from sensitivity import layerwise_sensitivity
    #     budget = layerwise_sensitivity(model, input_ids, cfg)
    #     model = prune_structured(model, budget)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
