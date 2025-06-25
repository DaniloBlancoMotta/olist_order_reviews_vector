# 🧠 Olist Reviews - API de Análise de Sentimentos com RAG

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/seu-usuario/olist-reviews)
[![Dataset](https://img.shields.io/badge/Dataset-Olist%20Brazilian%20E--commerce-orange.svg)](https://www.kaggle.com/olistbr/brazilian-ecommerce)

> **Transforme reviews de clientes em inteligência de produto** com nossa API completa de análise de sentimentos, combinando **busca semântica (RAG)** com **modelos de linguagem natural** para extrair insights valiosos dos feedbacks dos clientes.

## URL localhost Análise de Sentimento Dashboard: 

## 🚀 Características Principais

| Funcionalidade | Descrição | Status |
|----------------|-----------|--------|
| 🔍 **Busca Semântica** | Encontra reviews relevantes usando embeddings vetoriais FAISS | ✅ Ativo |
| 🧠 **Análise de Sentimentos** | Classifica sentimentos com modelos Hugging Face | ✅ Ativo |
| 📊 **Insights Estruturados** | Extrai pontos positivos, negativos e resumos | ✅ Ativo |
| ⚡ **API REST** | Interface FastAPI com documentação automática | ✅ Ativo |
| 🎯 **RAG (Retrieval-Augmented Generation)** | Combina busca e geração de texto | ✅ Ativo |
| 📈 **Análise em Tempo Real** | Processa reviews dinamicamente | ✅ Ativo |
| 🔧 **Configuração Flexível** | Suporte a variáveis de ambiente | ✅ Ativo |
| 📝 **Logging Avançado** | Sistema de logs configurável | ✅ Ativo |

## 📁 Estrutura do Projeto

```
olist-reviews/
├── 📁 src/olist_reviews/          
│   ├── 📁 api/                   
│   ├── 📁 sentiment/              
│   ├── 📁 rag/                  
│   ├── 📁 utils/                
│   └── config.py                  
├── 📁 data/                      
├── 📁 tests/                      
├── 📁 docs/                       
├── 📁 notebooks/                  
├── 📁 examples/                   
├── 📁 scripts/                    
├── run.py                         
├── setup.py                       
├── requirements.txt               
├── README.md                     
├── LICENSE                       
└── .gitignore                     


## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| **FastAPI** | 0.68+ | Framework web para APIs REST |
| **FAISS** | 1.7+ | Banco vetorial para busca semântica |
| **Sentence Transformers** | 2.0+ | Geração de embeddings vetoriais |
| **Transformers (Hugging Face)** | 4.0+ | Modelos de linguagem natural |
| **Pandas** | 1.3+ | Manipulação de dados tabulares |
| **NumPy** | 1.21+ | Computação numérica |
| **Uvicorn** | 0.15+ | Servidor ASGI para FastAPI |
| **Pydantic** | 1.8+ | Validação de dados e serialização |

## 📊 Dataset e Modelos

### 📈 Dataset Olist
- **Fonte**: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- **Tamanho**: 14MB (CSV)
- **Reviews**: +100k reviews de produtos brasileiros
- **Idioma**: Português brasileiro
- **Período**: 2016-2018

### 🤖 Modelos de IA
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Sentimentos**: `distilbert-base-uncased-finetuned-sst-2-english`
- **BERT**: `bert-base-uncased` (para consultas RAG)

## 🛠️ Instalação

### Pré-requisitos

- **Python**: 3.8 ou superior
- **RAM**: Mínimo 4GB (recomendado 8GB+)
- **Espaço**: 100MB para instalação + 77MB para dados
- **Sistema**: Windows, macOS ou Linux

### Setup Rápido (Recomendado)

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/olist-reviews.git
cd olist-reviews

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o setup completo
python run.py --full-setup
```

### Setup Manual (Avançado)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Construir índice FAISS (opcional - já incluído)
python scripts/build_index.py

# 3. Iniciar API
python run.py --start-api
```


### 2. Exemplo de Uso - Análise de Sentimentos

```python
import requests

# Analisar sentimentos de um produto
response = requests.post(
    "http://localhost:8000/analyze_sentiment",
    json={"product_id": "smartphone"}
)

result = response.json()
print(f" Sentimento: {result['predominant_sentiment']}")
print(f" Total reviews: {result['total_reviews']}")
print(f" Resumo: {result['summary']}")
print(f" Pontos positivos: {result['positive_points']}")
print(f" Pontos negativos: {result['negative_points']}")
```

### 3. Exemplo de Uso - Consulta RAG

```python
# Buscar reviews relevantes
response = requests.post(
    "http://localhost:8000/consultar_review",
    json={
        "query": "qualidade do produto e entrega",
        "top_k": 5,
        "similarity_threshold": 0.7


##  Endpoints da API

| Endpoint | Método | Descrição | Status |
|----------|--------|-----------|--------|
| `/` | GET | Informações da API | ✅ |
| `/health` | GET | Status de saúde | ✅ |
| `/stats` | GET | Estatísticas do sistema | ✅ |
| `/analyze_sentiment` | POST | Análise de sentimentos | ✅ |
| `/consultar_review` | POST | Consulta RAG | ✅ |

### 📋 Exemplo de Resposta - Análise de Sentimentos

```json
{
  "product_id": "smartphone",
  "predominant_sentiment": "positivo",
  "summary": "O produto é elogiado por sua qualidade e entrega eficiente, com raras reclamações de cor.",
  "positive_points": ["Entrega rápida", "Alta qualidade", "Bem embalado"],
  "negative_points": ["Manual não veio em português"],
  "representative_reviews": [
    {
      "text": "Excelente produto, entrega no dia seguinte!",
      "score": 5,
      "sentiment": "positivo"
    }
  ],
  "total_reviews": 150,
  "sentiment_distribution": {
    "positivo": 120,
    "neutro": 20,
    "negativo": 10
  }
}
```

📊 Distribuição das Notas
A maioria dos reviews tem nota 5 (57.8%)
Nota média é 4.09, indicando alta satisfação geral

Distribuição das notas:
Nota 5: 57,328 (57.8%)
Nota 4: 19,142 (19.3%)
Nota 1: 11,424 (11.5%)
Nota 3: 8,179 (8.2%)
Nota 2: 3,151 (3.2%)

⏰ Análise Temporal
Visualizações mostram a tendência diária e mensal dos reviews
Gráficos permitem identificar picos e quedas no volume de avaliações
Padrão sazonal por mês do ano

💬 Análise dos Comentários
41.3% dos reviews têm comentários (40,977)
58.7% são reviews sem comentários (58,247)

Características dos comentários:
Média: 68.6 caracteres
Mediana: 53.0 caracteres
Média de palavras: 11.7
Mediana de palavras: 9.0

🔤 Palavras Mais Frequentes
Top 10 palavras mais usadas:
produto: 18,428 vezes
prazo: 8,475 vezes
entrega: 6,528 vezes
antes: 5,626 vezes
chegou: 5,555 vezes
recebi: 5,274 vezes
bom: 4,607 vezes
recomendo: 4,337 vezes
entregue: 3,779 vezes
veio: 3,285 vezes


💡 Principais Insights
Alta Satisfação: A maioria dos clientes está muito satisfeita (77.1% dão notas 4 ou 5)
Foco em Logística: Palavras como "prazo", "entrega", "chegou" são muito frequentes
Comentários Concisos: A maioria dos comentários é relativamente curta (mediana de 9 palavras)
Feedback Positivo: Palavras como "bom", "recomendo", "excelente" aparecem com frequência
