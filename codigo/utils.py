"""Utilidades transversais: reprodutibilidade e persistência de resultados."""

from __future__ import annotations

import csv
import os
import random
from typing import Any, Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa as sementes de todas as fontes de aleatoriedade para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_result(results_dir: str, filename: str, row: Mapping[str, Any]) -> str:
    """Anexa uma linha de métricas a um CSV, criando cabeçalho na primeira escrita.

    Cada experimento (baseline, poda X a taxa Y, ...) grava uma linha; ao final,
    o CSV consolida a tabela que alimenta as figuras do Capítulo de Resultados.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path
