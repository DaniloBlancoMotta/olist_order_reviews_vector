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


## 📊 Distribuição das Notas

A análise das notas (`review_score`) revela um alto nível de satisfação entre os clientes.

* A **maioria dos reviews tem nota 5**, representando **57.8%** do total.
* A **nota média geral é de 4.09**, o que indica uma alta satisfação geral com os pedidos.

**Distribuição detalhada das notas:**
* **Nota 5**: 57.328 reviews (**57.8%**)
* **Nota 4**: 19.142 reviews (**19.3%**)
* **Nota 1**: 11.424 reviews (**11.5%**)
* **Nota 3**: 8.179 reviews (**8.2%**)
* **Nota 2**: 3.151 reviews (**3.2%**)

## ⏰ Análise Temporal dos Reviews

As visualizações temporais (gráficos diários e mensais da data de criação dos reviews) permitem observar a evolução do volume de avaliações ao longo do tempo.

* Os gráficos ajudam a identificar **picos e quedas** significativas no volume de avaliações, que podem estar associados a promoções, feriados ou outros eventos.
* Foi identificado um **padrão sazonal por mês do ano**, indicando variações cíclicas na atividade de reviews.

## 💬 Análise dos Comentários

Exploramos a presença e as características dos comentários textuais nos reviews.

* **41.3%** dos reviews possuem comentários (totalizando **40.977** reviews).
* **58.7%** dos reviews são apenas avaliações com notas, sem comentários adicionais (**58.247** reviews).

**Características dos comentários com texto:**
* **Média de caracteres**: 68.6
* **Mediana de caracteres**: 53.0
* **Média de palavras**: 11.7
* **Mediana de palavras**: 9.0

## 🔤 Palavras Mais Frequentes nos Comentários

Uma análise de frequência das palavras nos comentários (`review_comment_message`) revela os termos mais utilizados pelos clientes.

**Top 10 palavras mais usadas:**
1.  **produto**: 18.428 vezes
2.  **prazo**: 8.475 vezes
3.  **entrega**: 6.528 vezes
4.  **antes**: 5.626 vezes
5.  **chegou**: 5.555 vezes
6.  **recebi**: 5.274 vezes
7.  **bom**: 4.607 vezes
8.  **recomendo**: 4.337 vezes
9.  **entregue**: 3.779 vezes
10. **veio**: 3.285 vezes

## 💡 Principais Insights

Com base nas análises acima, podemos extrair as seguintes conclusões chave:

* **Alta Satisfação do Cliente**: A esmagadora maioria dos clientes está muito satisfeita, com **77.1%** das avaliações classificadas como **nota 4 ou 5**.
* **Foco na Logística e Entrega**: As palavras mais frequentes nos comentários ("prazo", "entrega", "chegou", "recebi", "entregue") indicam que a **logística e o cumprimento dos prazos** são aspectos cruciais e frequentemente mencionados pelos clientes.
* **Comentários Concisos**: A mediana de **9 palavras** por comentário sugere que a maioria dos clientes expressa seu feedback de forma direta e objetiva.
* **Feedback Positivo Dominante**: A recorrência de termos como "bom" e "recomendo" reforça o cenário de alta satisfação geral.

---


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


