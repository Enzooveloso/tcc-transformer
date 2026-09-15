"""Testes de corretude da poda — a validação que o capítulo de Desenvolvimento reivindica.

O teste central é o de **equivalência numérica**: remover fisicamente uma cabeça
de atenção (ou um neurônio MLP) tem de produzir exatamente a mesma saída que
apenas anular os pesos correspondentes no modelo de dimensões originais. Se as
duas coisas não coincidem, a remoção física está errada em algum detalhe de
indexação — e todo o capítulo de Resultados da poda estruturada cai com ela.

Por que a equivalência vale, no caso das cabeças: anular as fatias de Q, K e V
de uma cabeça (pesos *e* vieses) zera seu vetor de valores, de modo que sua
contribuição à saída da atenção é nula independentemente dos pesos de atenção
que o softmax produzir; anular as linhas correspondentes de ``c_proj`` fecha o
caminho. No MLP, ``gelu(0) = 0``, então anular a coluna de ``c_fc`` já basta.

Os testes usam um GPT-2 **minúsculo de pesos aleatórios** (2 camadas, 4 cabeças,
d_model 32), construído a partir da configuração e não baixado da plataforma
Hugging Face: o que está sob teste é a mecânica da poda, não os pesos
pré-treinados. Isso mantém a suíte rápida, offline e executável em CPU.

Uso:
    python test_poda.py        # executa tudo e imprime o relatório
    pytest test_poda.py        # se o pytest estiver disponível
"""

from __future__ import annotations

import copy

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prune_magnitude import prune_magnitude, prunable_sparsity
from prune_structured import (
    head_importances,
    mlp_importances,
    remove_heads,
    remove_mlp_neurons,
    select_structures,
)
from sensitivity import allocate_rates, prune_structured_per_layer

# Tolerâncias: a remoção física muda as formas das matrizes e, com elas, a
# ordem de soma do produto matricial — a igualdade é matemática, não bit a bit.
ATOL = 1e-5
RTOL = 1e-4

N_LAYER, N_HEAD, N_EMBD, N_INNER = 2, 4, 32, 64
SEQ = 7


def _modelo_teste(seed: int = 0) -> GPT2LMHeadModel:
    """GPT-2 diminuto de pesos aleatórios, em modo de avaliação e CPU."""
    torch.manual_seed(seed)
    cfg = GPT2Config(
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, n_inner=N_INNER,
        vocab_size=64, n_positions=32,
        # Sem dropout: qualquer aleatoriedade no forward invalidaria a comparação.
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _entrada(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 64, (1, SEQ))


@torch.no_grad()
def _logits(model, x) -> torch.Tensor:
    return model(x).logits


@torch.no_grad()
def _anular_cabecas(model, cabecas_por_camada: list[list[int]]) -> None:
    """Anula os pesos das cabeças indicadas, sem alterar as dimensões.

    Contraparte "contábil" de ``remove_heads``: o modelo continua com todas as
    cabeças, mas as podadas não contribuem em nada para a saída.
    """
    for camada, idxs in enumerate(cabecas_por_camada):
        if not idxs:
            continue
        attn = model.transformer.h[camada].attn
        head_dim = attn.head_dim
        split = attn.split_size
        for h in idxs:
            fatia = slice(h * head_dim, (h + 1) * head_dim)
            # Q, K e V ficam lado a lado em c_attn, cada um com `split` colunas.
            for bloco in range(3):
                cols = slice(bloco * split + fatia.start, bloco * split + fatia.stop)
                attn.c_attn.weight[:, cols] = 0.0
                attn.c_attn.bias[cols] = 0.0
            # c_proj consome as cabeças concatenadas: head_dim linhas cada.
            attn.c_proj.weight[fatia, :] = 0.0


@torch.no_grad()
def _anular_neuronios(model, neuronios_por_camada: list[list[int]]) -> None:
    """Anula os pesos dos neurônios MLP indicados, sem alterar as dimensões."""
    for camada, idxs in enumerate(neuronios_por_camada):
        if not idxs:
            continue
        mlp = model.transformer.h[camada].mlp
        for j in idxs:
            mlp.c_fc.weight[:, j] = 0.0
            mlp.c_fc.bias[j] = 0.0
            mlp.c_proj.weight[j, :] = 0.0


# ---------------------------------------------------------------------------
# Equivalência numérica: remoção física == anulamento
# ---------------------------------------------------------------------------

def test_remover_cabecas_equivale_a_anular():
    """Remover cabeças fisicamente dá a mesma saída que anulá-las."""
    x = _entrada()
    fisico, contabil = _modelo_teste(), _modelo_teste()

    # Mesmas cabeças nos dois modelos, escolhidas pelo critério de norma L1.
    cabecas = select_structures(head_importances(fisico), sparsity=0.5, scope="uniforme")
    assert any(cabecas), "o cenário de teste não removeu cabeça alguma"

    remove_heads(fisico, cabecas)
    _anular_cabecas(contabil, cabecas)

    torch.testing.assert_close(_logits(fisico, x), _logits(contabil, x),
                               atol=ATOL, rtol=RTOL)


def test_remover_neuronios_equivale_a_anular():
    """Remover neurônios MLP fisicamente dá a mesma saída que anulá-los."""
    x = _entrada()
    fisico, contabil = _modelo_teste(), _modelo_teste()

    neuronios = select_structures(mlp_importances(fisico), sparsity=0.5, scope="uniforme")
    assert any(neuronios), "o cenário de teste não removeu neurônio algum"

    remove_mlp_neurons(fisico, neuronios)
    _anular_neuronios(contabil, neuronios)

    torch.testing.assert_close(_logits(fisico, x), _logits(contabil, x),
                               atol=ATOL, rtol=RTOL)


def test_remover_ambos_equivale_a_anular():
    """A equivalência se mantém com cabeças e neurônios removidos juntos."""
    x = _entrada()
    fisico, contabil = _modelo_teste(), _modelo_teste()

    cabecas = select_structures(head_importances(fisico), 0.25, "uniforme")
    neuronios = select_structures(mlp_importances(fisico), 0.5, "uniforme")

    remove_heads(fisico, cabecas)
    remove_mlp_neurons(fisico, neuronios)
    _anular_cabecas(contabil, cabecas)
    _anular_neuronios(contabil, neuronios)

    torch.testing.assert_close(_logits(fisico, x), _logits(contabil, x),
                               atol=ATOL, rtol=RTOL)


def test_poda_nula_nao_altera_o_modelo():
    """Taxa 0 tem de ser identidade — controle interno de cada varredura."""
    x = _entrada()
    original, podado = _modelo_teste(), _modelo_teste()
    cabecas = select_structures(head_importances(podado), 0.0, "uniforme")
    neuronios = select_structures(mlp_importances(podado), 0.0, "uniforme")
    remove_heads(podado, cabecas)
    remove_mlp_neurons(podado, neuronios)
    torch.testing.assert_close(_logits(original, x), _logits(podado, x),
                               atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Contabilidade interna e salvaguardas
# ---------------------------------------------------------------------------

def test_contabilidade_da_atencao_fica_consistente():
    """``num_heads`` e ``split_size`` têm de acompanhar as formas dos tensores."""
    model = _modelo_teste()
    cabecas = select_structures(head_importances(model), 0.5, "uniforme")
    remove_heads(model, cabecas)

    for bloco in model.transformer.h:
        attn = bloco.attn
        assert attn.split_size == attn.num_heads * attn.head_dim
        # c_attn produz Q, K e V lado a lado: 3 x split_size colunas.
        assert attn.c_attn.weight.size(1) == 3 * attn.split_size
        assert attn.c_attn.bias.numel() == 3 * attn.split_size
        # c_proj recebe as cabeças concatenadas.
        assert attn.c_proj.weight.size(0) == attn.split_size


def test_guarda_preserva_uma_estrutura_por_camada():
    """Nem com taxa 100% uma camada pode ficar sem cabeça ou sem neurônio."""
    for escopo in ("uniforme", "global"):
        model = _modelo_teste()
        remove_heads(model, select_structures(head_importances(model), 1.0, escopo))
        remove_mlp_neurons(model, select_structures(mlp_importances(model), 1.0, escopo))
        for bloco in model.transformer.h:
            assert bloco.attn.num_heads >= 1, escopo
            assert bloco.mlp.c_fc.weight.size(1) >= 1, escopo
        # E o modelo continua executável (o fluxo residual não foi rompido).
        _logits(model, _entrada())


def test_estruturas_removidas_sao_as_de_menor_norma():
    """O critério de importância é a norma L1: sai a estrutura mais fraca."""
    model = _modelo_teste()
    scores = mlp_importances(model)
    remocoes = select_structures(scores, 0.5, "uniforme")
    for camada, idxs in enumerate(remocoes):
        removidas = scores[camada, idxs]
        mantidas_idx = sorted(set(range(scores.size(1))) - set(idxs))
        mantidas = scores[camada, mantidas_idx]
        assert removidas.max() <= mantidas.min()


# ---------------------------------------------------------------------------
# Poda por magnitude
# ---------------------------------------------------------------------------

def test_magnitude_atinge_a_esparsidade_alvo():
    """A esparsidade real sobre as matrizes podáveis bate com a solicitada."""
    for alvo in (0.1, 0.5, 0.9):
        for escopo in ("global", "uniforme"):
            model = _modelo_teste()
            prune_magnitude(model, alvo, scope=escopo)
            real = prunable_sparsity(model)
            assert abs(real - alvo) < 1e-3, (alvo, escopo, real)


def test_magnitude_preserva_as_dimensoes():
    """A poda não estruturada zera pesos sem remover nada da arquitetura."""
    original, podado = _modelo_teste(), _modelo_teste()
    prune_magnitude(podado, 0.5)
    for (n1, p1), (n2, p2) in zip(original.named_parameters(), podado.named_parameters()):
        assert n1 == n2 and p1.shape == p2.shape


def test_magnitude_zera_os_pesos_de_menor_modulo():
    """Os pesos sobreviventes têm de ter módulo maior que os zerados."""
    model = _modelo_teste()
    antes = model.transformer.h[0].mlp.c_fc.weight.detach().clone()
    prune_magnitude(model, 0.5, scope="uniforme")
    depois = model.transformer.h[0].mlp.c_fc.weight.detach()
    zerados, mantidos = depois == 0, depois != 0
    assert zerados.any() and mantidos.any()
    assert antes[zerados].abs().max() <= antes[mantidos].abs().min()


# ---------------------------------------------------------------------------
# Alocação por sensibilidade
# ---------------------------------------------------------------------------

def test_alocacao_respeita_o_orcamento():
    """A média das taxas, ponderada por parâmetros, iguala o orçamento alvo."""
    deltas = [4.0, 0.09, 0.17, 0.18, 0.22, 0.17, 0.14, 0.12, 0.12, 0.09, 0.14, 0.66]
    pesos = [1.0] * len(deltas)
    for alvo in (0.1, 0.3, 0.5, 0.7):
        taxas = allocate_rates(deltas, pesos, alvo, beta=1.0)
        media = sum(p * t for p, t in zip(pesos, taxas)) / sum(pesos)
        assert abs(media - alvo) < 1e-3, (alvo, media)
        assert all(t <= 0.95 + 1e-9 for t in taxas)


def test_alocacao_poda_menos_a_camada_mais_sensivel():
    """Camada com degradação maior recebe taxa menor — o cerne da estratégia."""
    deltas = [4.0, 0.09, 0.17, 0.66]
    taxas = allocate_rates(deltas, [1.0] * 4, 0.5, beta=1.0)
    assert taxas[0] == min(taxas)
    assert taxas[1] == max(taxas)


def test_beta_zero_degenera_na_poda_uniforme_estruturada():
    """Com beta=0 a alocação tem de reproduzir a varredura uniforme.

    Vale para a poda estruturada, em que a taxa por camada é aplicada do mesmo
    modo nos dois caminhos. NÃO vale para a magnitude: a alocação usa um limiar
    único por bloco (as 4 projeções juntas), enquanto o escopo uniforme da
    varredura usa um limiar por matriz — ver ``test_beta_zero_difere_na_magnitude``.
    """
    x = _entrada()
    deltas = [1.0, 2.0, 3.0][:N_LAYER]

    alocado = _modelo_teste()
    taxas = allocate_rates(deltas, [1.0] * N_LAYER, 0.5, beta=0.0)
    assert all(abs(t - taxas[0]) < 1e-9 for t in taxas), "beta=0 deve igualar as taxas"
    prune_structured_per_layer(alocado, taxas)

    uniforme = _modelo_teste()
    remove_heads(uniforme, select_structures(head_importances(uniforme), 0.5, "uniforme"))
    remove_mlp_neurons(uniforme,
                       select_structures(mlp_importances(uniforme), 0.5, "uniforme"))

    torch.testing.assert_close(_logits(alocado, x), _logits(uniforme, x),
                               atol=0.0, rtol=0.0)


def test_beta_zero_difere_na_magnitude():
    """Documenta a assimetria: na magnitude, beta=0 NÃO é a varredura uniforme.

    O limiar por bloco (alocação) e o limiar por matriz (varredura uniforme)
    produzem esparsidades iguais no agregado, mas distribuídas de forma
    diferente entre as 4 projeções de cada bloco. Este teste fixa esse
    comportamento para que ele não seja confundido com um defeito.
    """
    from sensitivity import prune_magnitude_per_layer

    alocado, uniforme = _modelo_teste(), _modelo_teste()
    taxas = [0.5] * N_LAYER
    prune_magnitude_per_layer(alocado, taxas)
    prune_magnitude(uniforme, 0.5, scope="uniforme")

    # Mesma esparsidade agregada...
    assert abs(prunable_sparsity(alocado) - prunable_sparsity(uniforme)) < 1e-3
    # ...mas não os mesmos pesos sobreviventes.
    w1 = alocado.transformer.h[0].mlp.c_fc.weight
    w2 = uniforme.transformer.h[0].mlp.c_fc.weight
    assert not torch.equal(w1 == 0, w2 == 0), (
        "se as máscaras coincidem, a assimetria documentada deixou de existir"
    )


# ---------------------------------------------------------------------------
# Máscara de esparsidade do fine-tuning
# ---------------------------------------------------------------------------

def test_mascara_sobrevive_a_um_passo_do_otimizador():
    """A máscara reaplicada mantém a esparsidade após um passo do AdamW.

    Reproduz em miniatura o cuidado descrito na metodologia: com momento e
    decaimento de pesos, zerar o gradiente não basta — os zeros voltariam a ser
    não nulos se a máscara não fosse reaplicada.
    """
    from finetune import capture_masks, reapply_masks

    model = _modelo_teste()
    prune_magnitude(model, 0.5, scope="uniforme")
    esparsidade_alvo = prunable_sparsity(model)
    masks = capture_masks(model)

    otimizador = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    x = _entrada()
    model.train()
    model(x, labels=x).loss.backward()
    otimizador.step()

    # Sem a máscara, o passo do otimizador ressuscita os pesos zerados.
    assert prunable_sparsity(model) < esparsidade_alvo - 1e-3, (
        "o cenário não é informativo: o otimizador não alterou os pesos zerados"
    )
    reapply_masks(model, masks)
    assert abs(prunable_sparsity(model) - esparsidade_alvo) < 1e-9


# ---------------------------------------------------------------------------
# Execução sem pytest
# ---------------------------------------------------------------------------

def main() -> int:
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    falhas = []
    for teste in testes:
        try:
            teste()
        except Exception as exc:  # noqa: BLE001 — relatório é o objetivo
            falhas.append((teste.__name__, exc))
            print(f"  FALHOU  {teste.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok      {teste.__name__}")

    print(f"\n{len(testes) - len(falhas)}/{len(testes)} testes passaram")
    if falhas:
        print("\nFalhas:")
        for nome, exc in falhas:
            print(f"  - {nome}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
