# 🧠 Olist Reviews - API de Análise de Sentimentos com RAG

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/seu-usuario/olist-reviews)
[![Dataset](https://img.shields.io/badge/Dataset-Olist%20Brazilian%20E--commerce-orange.svg)](https://www.kaggle.com/olistbr/brazilian-ecommerce)

> **Transforme reviews de clientes em inteligência de produto** com nossa API completa de análise de sentimentos, combinando **busca semântica (RAG)** com **modelos de linguagem natural** para extrair insights valiosos dos feedbacks dos clientes.

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
├── 📁 src/olist_reviews/          # Código fonte principal (pacote Python)
│   ├── 📁 api/                    # Endpoints FastAPI (1 arquivo)
│   ├── 📁 sentiment/              # Análise de sentimentos (1 arquivo)
│   ├── 📁 rag/                    # Busca semântica (1 arquivo)
│   ├── 📁 utils/                  # Utilitários (1 arquivo)
│   └── config.py                  # Configurações centralizadas
├── 📁 data/                       # Datasets e índices (3 arquivos, 77MB)
├── 📁 tests/                      # Testes automatizados (2 arquivos)
├── 📁 docs/                       # Documentação (4 arquivos)
├── 📁 notebooks/                  # Jupyter notebooks (3 arquivos)
├── 📁 examples/                   # Exemplos de uso (1 arquivo)
├── 📁 scripts/                    # Scripts utilitários (2 arquivos)
├── run.py                         # Script principal de execução
├── setup.py                       # Configuração do pacote
├── requirements.txt               # Dependências (18 pacotes)
├── README.md                      # Documentação principal
├── LICENSE                        # Licença MIT
└── .gitignore                     # Arquivos ignorados pelo Git
```

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

### Instalação como Pacote

```bash
# Instalação em modo desenvolvimento
pip install -e .

# Instalação com dependências de desenvolvimento
pip install -e .[dev]
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

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# Dataset
export DATASET_PATH="data/olist_order_reviews_dataset.csv"

# Modelos de IA
export MODEL_NAME="all-MiniLM-L6-v2"
export SENTIMENT_MODEL="distilbert-base-uncased-finetuned-sst-2-english"
export BERT_MODEL="bert-base-uncased"

# API

# RAG
export TOP_K_RESULTS="5"
export SIMILARITY_THRESHOLD="0.7"
export MAX_CONTEXT_LENGTH="512"

# Logging
export LOG_LEVEL="INFO"
```

### Arquivo de Configuração

As configurações podem ser ajustadas no arquivo `src/olist_reviews/config.py`:

```python
class Config:
    # Configurações do Dataset
    DATASET_PATH = "data/olist_order_reviews_dataset.csv"
    
    # Configurações dos Modelos
    MODEL_NAME = "all-MiniLM-L6-v2"
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Configurações da API
    API_HOST =
    API_PORT = 


```





