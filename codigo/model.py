"""Carregamento do modelo de referência (GPT-2 small) via Hugging Face."""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config


def load_model_and_tokenizer(cfg: Config):
    """Carrega os pesos pré-treinados do GPT-2 e o tokenizador correspondente.

    O modelo é colocado em modo de avaliação (``eval``) e no dispositivo alvo.
    Nenhuma poda é aplicada aqui: este é o baseline contra o qual todas as
    configurações podadas serão comparadas.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.to(cfg.device)
    model.eval()
    return model, tokenizer
