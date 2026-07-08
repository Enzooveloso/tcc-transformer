"""Poda por magnitude dos pesos — não estruturada (Han et al., 2015).

Primeira das duas estratégias clássicas da metodologia (granularidade fina).
A ideia é direta: os pesos de menor valor absoluto são considerados os menos
importantes e zerados. A poda é *não estruturada* porque zera conexões
individuais sem remover linhas/colunas inteiras — logo o formato denso das
matrizes (e, portanto, os FLOPs e o tempo de inferência em hardware comum)
permanece o mesmo. O que muda é o número de pesos **não-zerados**.

Esse contraste é justamente uma das perguntas empíricas do trabalho: em uma
GPU T4, a redução teórica de parâmetros da poda por magnitude tende a **não**
se traduzir em ganho real de energia, porque o zero continua sendo multiplicado.
A poda estruturada (``prune_structured.py``) é que ataca esse ponto.

Uso como biblioteca (gancho do ``run.py``):
    from prune_magnitude import prune_magnitude
    model = prune_magnitude(model, sparsity=0.5)          # global (padrão)
    model = prune_magnitude(model, sparsity=0.5, scope="uniforme")

Uso como script (varredura completa perplexity x esparsidade):
    python prune_magnitude.py
    python prune_magnitude.py --scope uniforme --sparsities 0 0.3 0.5 0.7 0.9
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
import torch.nn as nn

from config import Config
from data import load_encodings
from energy import track_energy
from eval import evaluate_all
from model import load_model_and_tokenizer
from utils import append_result, set_seed

# O GPT-2 implementa as projeções de atenção e MLP como camadas ``Conv1D``
# (equivalente a uma linear com pesos transpostos), não como ``nn.Linear``.
try:  # local do símbolo varia entre versões do transformers
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    from transformers.modeling_utils import Conv1D  # type: ignore


def _prunable_named_weights(model):
    """Seleciona as matrizes de peso elegíveis para poda.

    Poda-se apenas os pesos das projeções lineares dos blocos Transformer
    (``c_attn``, ``c_proj``, ``mlp.c_fc``, ``mlp.c_proj``). Ficam de fora, por
    convenção da literatura:
      - as matrizes de *embedding* (``wte``/``wpe``), que codificam o vocabulário;
      - a cabeça de linguagem (``lm_head``), que compartilha pesos com ``wte``;
      - vieses e parâmetros de LayerNorm, que são baratos e sensíveis.
    """
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            if name.endswith("lm_head"):
                continue  # amarrada a wte: podá-la corromperia os embeddings
            prunable.append((name, module.weight))
    return prunable


def _magnitude_threshold(scores: torch.Tensor, sparsity: float) -> float:
    """Devolve o limiar de magnitude abaixo do qual os pesos são zerados.

    ``scores`` é o vetor de magnitudes (|peso|). Zeramos os ``sparsity``% de
    menor magnitude, o que equivale a manter apenas os pesos estritamente
    acima do k-ésimo menor valor.
    """
    if sparsity <= 0:
        return -1.0  # nada a podar: toda magnitude >= 0 fica acima de -1
    k = int(sparsity * scores.numel())
    if k <= 0:
        return -1.0
    if k >= scores.numel():
        return float("inf")  # esparsidade total: tudo é zerado
    return torch.kthvalue(scores, k).values.item()


@torch.no_grad()
def prune_magnitude(model, sparsity: float, scope: str = "global"):
    """Aplica poda por magnitude, in-place, e devolve o próprio modelo.

    ``sparsity`` é a fração de pesos podáveis a zerar (0.0–1.0).

    ``scope`` define como o limiar é escolhido:
      - ``"global"``   — um único limiar para todos os pesos podáveis juntos;
                         camadas mais robustas cedem mais pesos que as sensíveis.
      - ``"uniforme"`` — o mesmo percentual em cada camada, isoladamente; é a
                         poda *ingênua* que serve de contraponto à análise de
                         sensibilidade por camada (contribuição central do TCC).
    """
    prunable = _prunable_named_weights(model)
    if not prunable:
        raise RuntimeError("Nenhuma camada podável encontrada no modelo.")

    if scope == "global":
        scores = torch.cat([w.abs().flatten() for _, w in prunable])
        threshold = _magnitude_threshold(scores, sparsity)
        for _, w in prunable:
            w.mul_((w.abs() > threshold).to(w.dtype))
    elif scope == "uniforme":
        for _, w in prunable:
            threshold = _magnitude_threshold(w.abs().flatten(), sparsity)
            w.mul_((w.abs() > threshold).to(w.dtype))
    else:
        raise ValueError(f"scope inválido: {scope!r} (use 'global' ou 'uniforme')")

    return model


@torch.no_grad()
def prunable_sparsity(model) -> float:
    """Esparsidade *real* obtida sobre as matrizes podáveis (fração de zeros).

    Difere da esparsidade global de ``count_parameters`` porque desconsidera
    embeddings e vieses — é a fração que de fato foi zerada pela poda.
    """
    prunable = _prunable_named_weights(model)
    total = sum(w.numel() for _, w in prunable)
    zeros = sum(int((w == 0).sum().item()) for _, w in prunable)
    return zeros / total if total else 0.0


def parse_args() -> tuple[Config, str, list[float]]:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Varredura de poda por magnitude (GPT-2)."
    )
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--dataset-config", default=cfg.dataset_config,
                        help="ex.: wikitext-2-raw-v1 ou wikitext-103-raw-v1")
    parser.add_argument("--scope", default="global", choices=["global", "uniforme"],
                        help="global (limiar único) ou uniforme (mesmo %% por camada)")
    parser.add_argument("--sparsities", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="frações de esparsidade a avaliar")
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
    return cfg, args.scope, args.sparsities


def evaluate_at_sparsity(cfg: Config, tokenizer, input_ids, sparsity: float,
                         scope: str) -> dict:
    """Carrega um modelo limpo, poda-o à ``sparsity`` alvo e coleta as métricas.

    O modelo é recarregado do zero a cada nível para que os experimentos sejam
    independentes (evita poda cumulativa: 0.5 aplicado sobre 0.3 e assim por
    diante).
    """
    model, _ = load_model_and_tokenizer(cfg)
    prune_magnitude(model, sparsity, scope=scope)

    metrics = {
        "experimento": f"magnitude_{scope}_s{int(round(sparsity * 100)):02d}",
        "modelo": cfg.model_name,
        "dataset": cfg.dataset_config,
        "escopo": scope,
        "esparsidade_alvo": sparsity,
        "esparsidade_real": prunable_sparsity(model),
    }
    with track_energy(cfg, metrics):
        metrics.update(evaluate_all(model, input_ids, cfg))
    return metrics


def main() -> None:
    cfg, scope, sparsities = parse_args()
    set_seed(cfg.seed)

    print(f"[magnitude] dispositivo: {cfg.device}")
    print(f"[magnitude] modelo: {cfg.model_name} | dataset: {cfg.dataset_config}")
    print(f"[magnitude] escopo: {scope} | esparsidades: {sparsities}")

    # Tokeniza o conjunto de avaliação uma única vez (não depende da poda).
    _, tokenizer = load_model_and_tokenizer(cfg)
    input_ids = load_encodings(cfg, tokenizer)
    print(f"[magnitude] tokens de avaliação: {input_ids.size(1):,}")

    for sparsity in sparsities:
        metrics = evaluate_at_sparsity(cfg, tokenizer, input_ids, sparsity, scope)
        print(f"\n--- esparsidade alvo {sparsity:.0%} "
              f"(real {metrics['esparsidade_real']:.2%}) ---")
        print(f"  perplexity: {metrics['perplexity']:.4f} | "
              f"nao_zerados: {metrics['parametros_nao_zerados']:,}")
        path = append_result(cfg.results_dir, "magnitude.csv", metrics)

    print(f"\n[magnitude] varredura anexada em: {path}")


if __name__ == "__main__":
    main()
