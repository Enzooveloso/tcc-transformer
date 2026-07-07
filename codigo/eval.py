"""Métricas de avaliação: desempenho preditivo e custo computacional.

Cobre as duas primeiras dimensões da metodologia:
  - Desempenho preditivo: perplexity;
  - Custo computacional: parâmetros, FLOPs, tempo de inferência, tamanho.
A terceira dimensão (impacto ambiental) fica em ``energy.py``.
"""

from __future__ import annotations

import time

import torch

from config import Config


@torch.no_grad()
def compute_perplexity(model, input_ids: torch.Tensor, cfg: Config) -> float:
    """Calcula a perplexity por janela deslizante (estratégia padrão para GPT-2).

    O texto é percorrido em janelas de ``max_length`` com passo ``stride``. Em
    cada janela, apenas os tokens ainda não avaliados contribuem para a perda
    (os demais entram como contexto, com alvo -100 = ignorado). A perplexity é
    a exponencial da perda média por token: quanto menor, melhor a modelagem.
    """
    device = cfg.device
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, cfg.stride):
        end = min(begin + cfg.max_length, seq_len)
        trg_len = end - prev_end  # tokens realmente avaliados nesta janela

        window = input_ids[:, begin:end].to(device)
        targets = window.clone()
        targets[:, :-trg_len] = -100  # ignora o contexto já contabilizado

        outputs = model(window, labels=targets)

        # outputs.loss é a média sobre os (trg_len - 1) tokens previstos.
        num_valid = trg_len - 1
        nll_sum += outputs.loss.item() * num_valid
        n_tokens += num_valid

        prev_end = end
        if end == seq_len:
            break

    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))


def count_parameters(model) -> dict:
    """Conta parâmetros totais e não-zerados.

    A distinção importa para a poda: a não estruturada zera pesos sem removê-los,
    de modo que 'total' permanece constante e apenas 'nao_zerados' cai.
    """
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return {"parametros_total": total, "parametros_nao_zerados": nonzero}


def model_size_mb(model) -> float:
    """Tamanho do modelo em memória (MB), somando bytes de todos os parâmetros."""
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_total / (1024 ** 2)


@torch.no_grad()
def inference_time_ms(model, cfg: Config) -> dict:
    """Mede o tempo de uma inferência sobre uma entrada representativa.

    Faz execuções de aquecimento (descartadas) e depois repetições cronometradas,
    reportando média e desvio-padrão em milissegundos para mitigar a variância.
    """
    device = cfg.device
    dummy = torch.randint(
        low=0, high=model.config.vocab_size,
        size=(1, cfg.inference_seq_len), device=device,
    )

    for _ in range(cfg.inference_warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.inference_repeats):
        start = time.perf_counter()
        model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    t = torch.tensor(times)
    return {
        "tempo_inferencia_ms_media": float(t.mean()),
        "tempo_inferencia_ms_desvio": float(t.std()),
    }


def compute_flops(model, cfg: Config) -> float:
    """Estima os FLOPs de uma inferência (forward pass).

    Tenta a contagem exata via ``thop`` (análise do grafo); se a biblioteca não
    estiver disponível ou falhar em algum operador do Transformer, recai sobre a
    estimativa analítica clássica: ~2 x (parâmetros não-embedding) por token.
    """
    seq_len = cfg.inference_seq_len
    dummy = torch.randint(
        low=0, high=model.config.vocab_size,
        size=(1, seq_len), device=cfg.device,
    )

    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(2 * macs)  # 1 MAC = 2 FLOPs
    except Exception:
        # Fallback analítico: exclui a matriz de embeddings da contagem.
        non_embedding = sum(
            p.numel() for n, p in model.named_parameters() if "wte" not in n and "wpe" not in n
        )
        return float(2 * non_embedding * seq_len)


def evaluate_all(model, input_ids: torch.Tensor, cfg: Config) -> dict:
    """Agrega todas as métricas de qualidade e custo em um único dicionário."""
    metrics = {}
    metrics["perplexity"] = compute_perplexity(model, input_ids, cfg)
    metrics.update(count_parameters(model))
    metrics["tamanho_mb"] = model_size_mb(model)
    metrics["flops"] = compute_flops(model, cfg)
    metrics.update(inference_time_ms(model, cfg))
    return metrics
