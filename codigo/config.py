"""Configuração central da pipeline de poda neural (GPT-2 small).

Todas as decisões que a metodologia remete ao Capítulo de Desenvolvimento
(modelo, dataset, hardware) ficam centralizadas aqui, para que os experimentos
sejam reprodutíveis e fáceis de variar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _default_device() -> str:
    """Usa GPU (T4 no Kaggle) quando disponível; cai para CPU em testes locais."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # --- Modelo de referência (baseline) -----------------------------------
    model_name: str = "gpt2"  # GPT-2 small (124M) na plataforma Hugging Face

    # --- Dataset de avaliação ----------------------------------------------
    # Padrão WikiText-2; parametrizável para WikiText-103 sem alterar o código.
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "test"

    # --- Cálculo de perplexity (janela deslizante) -------------------------
    # max_length limitado por n_positions do GPT-2 (1024); stride controla a
    # sobreposição entre janelas — quanto menor, mais preciso e mais lento.
    max_length: int = 1024
    stride: int = 512

    # --- Medição de tempo de inferência ------------------------------------
    inference_warmup: int = 3     # execuções descartadas (aquecimento da GPU)
    inference_repeats: int = 20   # execuções cronometradas (média +/- desvio)
    inference_seq_len: int = 512  # comprimento da entrada representativa

    # --- Reprodutibilidade --------------------------------------------------
    seed: int = 42

    # --- Ambiente -----------------------------------------------------------
    device: str = field(default_factory=_default_device)

    # --- Saída --------------------------------------------------------------
    results_dir: str = "resultados"        # CSVs gerados pela pipeline
    energy_enabled: bool = True            # liga/desliga CodeCarbon
