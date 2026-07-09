# Graph Report - .  (2026-07-08)

## Corpus Check
- Corpus is ~12,690 words - fits in a single context window. You may not need a graph.

## Summary
- 93 nodes · 159 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.8)
- Token cost: 71,845 input · 3,000 output

## Community Hubs (Navigation)
- Métricas de Avaliação (eval.py)
- Poda por Magnitude
- Orquestração da Pipeline
- Estratégias de Poda & Literatura
- Modelo & Estratégias de Otimização
- Carregamento de Dados (WikiText)
- Configuração & Infraestrutura
- Medição Energética (CodeCarbon)
- Capa UFVJM (Institucional)
- Capa UFVJM (Template)
- PyTorch

## God Nodes (most connected - your core abstractions)
1. `Config` - 18 edges
2. `Pipeline de Poda Neural GPT-2` - 14 edges
3. `evaluate_all()` - 11 edges
4. `evaluate_at_sparsity()` - 9 edges
5. `run_baseline()` - 9 edges
6. `Compressão de LLMs via Poda Neural (TCC)` - 9 edges
7. `main()` - 7 edges
8. `load_encodings()` - 6 edges
9. `load_model_and_tokenizer()` - 6 edges
10. `track_energy()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Análise de Sensibilidade por Camada` --semantically_similar_to--> `Knowledge Distillation (KD)`  [INFERRED] [semantically similar]
  README.md → codigo/README.md
- `transformers` --conceptually_related_to--> `Hugging Face (plataforma)`  [INFERRED]
  codigo/requirements.txt → README.md
- `main()` --calls--> `load_encodings()`  [INFERRED]
  codigo/prune_magnitude.py → codigo/data.py
- `run_baseline()` --calls--> `load_encodings()`  [INFERRED]
  codigo/run.py → codigo/data.py
- `evaluate_at_sparsity()` --calls--> `track_energy()`  [INFERRED]
  codigo/prune_magnitude.py → codigo/energy.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Estratégias de Poda e Otimização Investigadas** — readme_magnitude_pruning, readme_structured_pruning, readme_layer_sensitivity_analysis [EXTRACTED 1.00]
- **Fluxo do Pipeline Baseline (Estágio 0)** — codigo_run, codigo_data, codigo_model, codigo_eval, codigo_energy [INFERRED 0.75]
- **Métricas do Compromisso Qualidade x Eficiência** — codigo_readme_quality_efficiency_tradeoff, readme_perplexity, readme_energy_measurement [EXTRACTED 1.00]

## Communities (11 total, 1 thin omitted)

### Community 0 - "Métricas de Avaliação (eval.py)"
Cohesion: 0.19
Nodes (16): Config, compute_flops(), compute_perplexity(), count_parameters(), evaluate_all(), inference_time_ms(), model_size_mb(), Tensor (+8 more)

### Community 1 - "Poda por Magnitude"
Cohesion: 0.18
Nodes (16): load_model_and_tokenizer(), Carrega os pesos pré-treinados do GPT-2 e o tokenizador correspondente.      O m, evaluate_at_sparsity(), _magnitude_threshold(), main(), parse_args(), _prunable_named_weights(), prunable_sparsity() (+8 more)

### Community 2 - "Orquestração da Pipeline"
Cohesion: 0.19
Nodes (12): Any, Estágio 0: Baseline, main(), parse_args(), Orquestrador da pipeline de poda neural — GPT-2 small.  Estágio 0 (implementado), Carrega o baseline, avalia e devolve o dicionário de métricas., run_baseline(), append_result() (+4 more)

### Community 3 - "Estratégias de Poda & Literatura"
Cohesion: 0.22
Nodes (11): prune_structured.py, Compromisso Qualidade x Eficiência, Trabalho Anterior CNN (ResNet-50/CIFAR-10), CodeCarbon (energia, kWh e CO2), Mensuração Energética, Han et al., 2015, Li et al., 2017, Poda por Magnitude (Han et al., 2015) (+3 more)

### Community 4 - "Modelo & Estratégias de Otimização"
Cohesion: 0.27
Nodes (9): finetune.py, Carregamento do modelo de referência (GPT-2 small) via Hugging Face., Knowledge Distillation (KD), Pipeline de Poda Neural GPT-2, transformers, sensitivity.py, GPT-2 small (124M), Hugging Face (plataforma) (+1 more)

### Community 5 - "Carregamento de Dados (WikiText)"
Cohesion: 0.29
Nodes (6): load_encodings(), Tensor, Carregamento e tokenização do WikiText para avaliação de perplexity., Carrega o split de avaliação do WikiText e o tokeniza como um único fluxo., datasets, WikiText

### Community 6 - "Configuração & Infraestrutura"
Cohesion: 0.40
Nodes (4): _default_device(), Configuração central da pipeline de poda neural (GPT-2 small).  Todas as decisõe, Usa GPU (T4 no Kaggle) quando disponível; cai para CPU em testes locais., Kaggle Notebooks (GPU Tesla T4)

### Community 7 - "Medição Energética (CodeCarbon)"
Cohesion: 0.50
Nodes (3): Medição de impacto ambiental (energia e CO2) via CodeCarbon.  Terceira dimensão, Context manager que mede a energia consumida no bloco envolvido.      Uso:, track_energy()

### Community 8 - "Capa UFVJM (Institucional)"
Cohesion: 0.67
Nodes (4): UFVJM Thesis Cover Template, Green and Blue Decorative Background, Universidade Federal dos Vales do Jequitinhonha e Mucuri, UFVJM Institutional Logo

### Community 9 - "Capa UFVJM (Template)"
Cohesion: 0.67
Nodes (3): UFVJM Thesis Cover Template (Decorative), TCC Monograph Cover Page, UFVJM Visual Identity (Blue/Green Graphics)

## Knowledge Gaps
- **11 isolated node(s):** `Han et al., 2015`, `Li et al., 2017`, `Trabalho Anterior CNN (ResNet-50/CIFAR-10)`, `Estágio 0: Baseline`, `finetune.py` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Métricas de Avaliação (eval.py)` to `Poda por Magnitude`, `Orquestração da Pipeline`, `Modelo & Estratégias de Otimização`, `Carregamento de Dados (WikiText)`, `Configuração & Infraestrutura`, `Medição Energética (CodeCarbon)`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `Pipeline de Poda Neural GPT-2` connect `Modelo & Estratégias de Otimização` to `Métricas de Avaliação (eval.py)`, `Poda por Magnitude`, `Orquestração da Pipeline`, `Estratégias de Poda & Literatura`, `Carregamento de Dados (WikiText)`, `Configuração & Infraestrutura`, `Medição Energética (CodeCarbon)`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Why does `evaluate_all()` connect `Métricas de Avaliação (eval.py)` to `Poda por Magnitude`, `Orquestração da Pipeline`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `evaluate_all()` (e.g. with `evaluate_at_sparsity()` and `run_baseline()`) actually correct?**
  _`evaluate_all()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `evaluate_at_sparsity()` (e.g. with `track_energy()` and `evaluate_all()`) actually correct?**
  _`evaluate_at_sparsity()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `run_baseline()` (e.g. with `load_encodings()` and `track_energy()`) actually correct?**
  _`run_baseline()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Configuração central da pipeline de poda neural (GPT-2 small).  Todas as decisõe`, `Usa GPU (T4 no Kaggle) quando disponível; cai para CPU em testes locais.`, `Carregamento e tokenização do WikiText para avaliação de perplexity.` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._