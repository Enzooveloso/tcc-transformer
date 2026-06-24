# TCC — Compressão de LLMs via Poda Neural

Trabalho de Conclusão de Curso de Enzo Veloso, graduando em Sistemas de Informação na UFVJM, sob orientação do Prof. Dr. Marcelo Ferreira Rego, no escopo do projeto FAPEMIG BIS-00587-25.

## Tema

Estudo e desenvolvimento de estratégias de compressão de redes neurais para redução do consumo de energia em LLMs (Large Language Models).

## Abordagem

- **Modelo de referência:** GPT-2 small (124M parâmetros) obtido via plataforma Hugging Face
- **Tarefa de avaliação:** modelagem de linguagem em WikiText
- **Métrica principal:** perplexity
- **Estratégias de poda investigadas:**
  - Poda por magnitude (Han et al., 2015) — granularidade fina
  - Poda estruturada (Li et al., 2017) — granularidade grossa
- **Estratégia de otimização proposta:** análise de sensibilidade por camada
- **Infraestrutura experimental:** Kaggle Notebooks (GPU Tesla T4)
- **Mensuração energética:** biblioteca CodeCarbon (kWh e CO₂)

## Estrutura do Repositório

```
escrita/        → Monografia em LaTeX (ABNTeX2)
codigo/         → Pipeline de poda em PyTorch (em desenvolvimento)
```

## Compilação

Pré-requisitos: TeX Live (com `pdflatex`).

```bash
cd escrita
pdflatex principal.tex && pdflatex principal.tex
```

## Trabalho Anterior

O projeto começou explorando redes neurais convolucionais (ResNet-50/CIFAR-10). Esse trabalho está mantido em repositório separado ([artigo-cnn](https://github.com/SEU_USUARIO/artigo-cnn)) e será submetido como artigo científico independente.

## Contato

- **Autor:** Enzo Veloso
- **Orientador:** Prof. Dr. Marcelo Ferreira Rego — UFVJM
