# codigo/

Pipeline de poda neural para **GPT-2 small** (124M), em PyTorch, com pesos
pré-treinados obtidos via plataforma Hugging Face.

**Alvo de execução:** Kaggle Notebooks (GPU Tesla T4). Roda também localmente
em CPU para testes rápidos (o dispositivo é detectado automaticamente).

## Objetivo experimental

Medir o compromisso **qualidade × eficiência** (perplexity vs. FLOPs/energia)
sob diferentes estratégias de poda, e testar se a poda **otimizada** (análise de
sensibilidade por camada) melhora esse compromisso frente à poda ingênua.
Ver o relatório de tese no capítulo de Metodologia.

## Estrutura

| Arquivo | Papel | Estado |
|---------|-------|--------|
| `config.py` | Configuração central (modelo, dataset, hardware, seeds) | ✅ |
| `utils.py` | Reprodutibilidade (seeds) e escrita de resultados (CSV) | ✅ |
| `data.py` | Carrega e tokeniza o WikiText | ✅ |
| `model.py` | Carrega o GPT-2 small (baseline) | ✅ |
| `eval.py` | Perplexity + custo (parâmetros, FLOPs, tempo, tamanho) | ✅ |
| `energy.py` | Energia e CO₂ via CodeCarbon | ✅ |
| `run.py` | Orquestrador — **Estágio 0: baseline** | ✅ |
| `prune_magnitude.py` | Poda por magnitude (Han, 2015) | ⏳ próximo |
| `prune_structured.py` | Poda estruturada de cabeças/MLP (Li, 2017) | ⏳ |
| `sensitivity.py` | Análise de sensibilidade por camada | ⏳ |
| `finetune.py` | Fine-tuning pós-poda (one-shot fica preparado) | ⏳ |

## Como rodar (Estágio 0 — baseline)

```bash
pip install -r requirements.txt
python run.py                                        # WikiText-2 (padrão)
python run.py --dataset-config wikitext-103-raw-v1   # trocar dataset
python run.py --no-energy                            # sem medição de energia
```

O baseline gera a primeira linha de `resultados/resultados.csv`, com o vetor
completo de métricas contra o qual todas as configurações podadas serão comparadas.

## Ordem de desenvolvimento

1. **Baseline + perplexity** (feito) — valida dados, modelo e métrica.
2. Poda por magnitude (não estruturada) end-to-end.
3. Poda estruturada (cabeças de atenção / neurônios MLP).
4. Otimizações — sensibilidade por camada (contribuição central), KD.
