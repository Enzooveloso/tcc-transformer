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
| `prune_magnitude.py` | Poda por magnitude (Han, 2015) | ✅ |
| `prune_structured.py` | Poda estruturada de cabeças/MLP (Li, 2017) | ✅ |
| `sensitivity.py` | Análise de sensibilidade por camada | ✅ |
| `finetune.py` | Fine-tuning de recuperação pós-poda | ✅ |

## Como rodar (Estágio 0 — baseline)

```bash
pip install -r requirements.txt
python run.py                                        # WikiText-2 (padrão)
python run.py --dataset-config wikitext-103-raw-v1   # trocar dataset
python run.py --no-energy                            # sem medição de energia
```

O baseline gera a primeira linha de `resultados/resultados.csv`, com o vetor
completo de métricas contra o qual todas as configurações podadas serão comparadas.

## Como rodar as varreduras de poda

```bash
# Magnitude (não estruturada) — grava em resultados/magnitude.csv
python prune_magnitude.py                    # escopo global (padrão)
python prune_magnitude.py --scope uniforme

# Estruturada (cabeças/neurônios MLP) — grava em resultados/estruturada.csv
python prune_structured.py                   # ambos os alvos, escopo global
python prune_structured.py --scope uniforme
python prune_structured.py --alvo cabecas    # só cabeças de atenção
python prune_structured.py --alvo mlp        # só neurônios MLP
```

Cada nível de esparsidade recarrega um modelo limpo (sem poda cumulativa).
Na estruturada, `--sparsities` é a fração de **estruturas** removidas por alvo;
a fração de parâmetros correspondente sai na coluna `esparsidade_real`.

## Como rodar a análise de sensibilidade (contribuição central)

```bash
# Fase 1 (perfil por camada) + Fase 2 (varredura com alocação otimizada)
python sensitivity.py                          # as duas estratégias

# Fases separadas / opções
python sensitivity.py --fase perfil            # só o perfil (sensibilidade_perfil.csv)
python sensitivity.py --fase varredura         # só a varredura (lê o perfil do CSV)
python sensitivity.py --estrategias estruturada --taxa-sonda 0.3
python sensitivity.py --fase varredura --beta 0.5   # ablação da alocação
```

O perfil poda **uma camada por vez** à taxa-sonda e mede a degradação de
perplexity; a varredura converte o perfil em taxas por camada (robustas cedem
mais) e grava em `resultados/sensibilidade.csv` — o competidor "otimizado"
contra os escopos `global` e `uniforme` das varreduras ingênuas.

O quanto o perfil é levado a sério é controlado por `--beta` (taxa ∝
robustez^β): `0` recai na poda uniforme, `1` (padrão) usa o inverso puro da
degradação medida. Rodar a varredura com dois betas sobre o **mesmo** perfil dá
a ablação da função de alocação sem refazer a fase 1, que é a cara.

Rode a fase 1 primeiro e guarde o `sensibilidade_perfil.csv`: a varredura é
refeita a partir dele a qualquer momento, o perfil não. O log da varredura
informa quantas camadas ficaram no teto de poda a cada orçamento — muitas
camadas no teto indicam alocação concentrada em poucas camadas.

## Como rodar o fine-tuning de recuperação

Todas as varreduras acima são *one-shot* (poda e mede, sem retreinar). O
`finetune.py` fecha essa lacuna: poda, **retreina** no split de treino do
corpus e reavalia, como fazem Han (2015) e Li (2017).

```bash
# CONTROLE DENSO — rode sempre, na mesma sessão que os demais
python finetune.py --estrategia nenhuma

# As três estratégias
python finetune.py --estrategia magnitude --scope uniforme --sparsities 0.3 0.5
python finetune.py --estrategia estruturada --scope global --sparsities 0.1 0.3
python finetune.py --estrategia sensibilidade --sens-base estruturada --beta 1.0

# Orçamento de treino e diagnóstico
python finetune.py --estrategia nenhuma --max-steps 200 --eval-every 50
python finetune.py --estrategia magnitude --limit-train-blocks 20 --max-steps 5
```

Grava em `resultados/finetune.csv` (uma linha por configuração, com
`perplexity_pre`, `perplexity` e `recuperacao_log`) e, com `--eval-every`, a
curva de recuperação em `resultados/finetune_curva.csv`.

**O controle denso não é opcional.** O GPT-2 não foi pré-treinado no WikiText-2,
então treinar o modelo **denso** nesse corpus já baixa a perplexity por
adaptação de domínio. Sem essa linha é impossível separar "recuperei o dano da
poda" de "adaptei o modelo ao corpus", e toda a leitura do capítulo fica
comprometida.

**Preservação da esparsidade.** Na poda não estruturada os pesos são só
zerados, e o gradiente os ressuscitaria no primeiro passo do otimizador. Uma
máscara binária fixa é capturada após a poda e reaplicada depois de **cada**
passo (zerar o gradiente não basta: momento e *weight decay* mexem no peso
mesmo com gradiente nulo). A coluna `esparsidade_pos` registra a esparsidade
ao final do treino e deve bater com `esparsidade_real` — se não bater, o log
avisa e a linha é inválida. Na poda estruturada não há máscara: as estruturas
já foram removidas fisicamente.

**Semente passa a importar.** As varreduras one-shot são determinísticas
(mesma perplexity até o último dígito entre sessões). O fine-tuning não é
— ordem dos lotes e dropout —, então aqui repetições com `--seed` distintas
passam a fazer sentido.

**Energia do treino.** O custo energético do retreinamento é medido à parte,
em `energia_treino_kwh`: é um custo pago uma vez, que precisa ser amortizado
contra a economia por inferência para a conta energética do trabalho fechar.

## Ordem de desenvolvimento

1. **Baseline + perplexity** (feito) — valida dados, modelo e métrica.
2. Poda por magnitude (não estruturada) end-to-end.
3. Poda estruturada (cabeças de atenção / neurônios MLP).
4. Otimizações — sensibilidade por camada (contribuição central), KD.
5. Fine-tuning de recuperação, aplicado às três estratégias.
