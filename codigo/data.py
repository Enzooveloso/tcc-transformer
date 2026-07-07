"""Carregamento e tokenização do WikiText para avaliação de perplexity."""

from __future__ import annotations

import torch
from datasets import load_dataset

from config import Config


def load_encodings(cfg: Config, tokenizer) -> torch.Tensor:
    """Carrega o split de avaliação do WikiText e o tokeniza como um único fluxo.

    A perplexity de um modelo de linguagem é calculada sobre o texto contínuo:
    concatena-se todo o conjunto em uma única sequência de tokens, que depois é
    percorrida por janelas deslizantes (ver ``eval.compute_perplexity``).

    Retorna um tensor 1 x N com os IDs dos tokens.
    """
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)

    # As linhas do WikiText são unidas com quebra dupla, preservando a separação
    # entre parágrafos/artigos tal como o corpus original.
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids
