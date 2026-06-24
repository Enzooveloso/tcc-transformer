# codigo/

Esta pasta receberá o pipeline de poda neural para GPT-2 small, a ser desenvolvido em PyTorch.

O pipeline incluirá:
- Carregamento do modelo GPT-2 via Hugging Face
- Avaliação de perplexity em WikiText
- Poda por magnitude (não estruturada)
- Poda estruturada (cabeças de atenção / neurônios MLP)
- Análise de sensibilidade por camada
- Mensuração energética via CodeCarbon
