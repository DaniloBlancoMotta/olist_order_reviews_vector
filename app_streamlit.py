#!/usr/bin/env python3
"""
Dashboard de Análise de Sentimento e Tópicos - Olist Reviews
Aplicação Streamlit para insights rápidos sobre percepção do cliente
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Adiciona o diretório src ao path para importar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importações dos módulos do projeto
from olist_reviews.config import Config
from olist_reviews.rag.vector_store import VectorStore
from olist_reviews.sentiment.sentiment_analyzer import SentimentAnalyzer

# Configuração da página
st.set_page_config(
    page_title="Dashboard Olist Reviews",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download dos recursos do NLTK (executar apenas uma vez)
@st.cache_resource
def download_nltk_resources():
    """Download dos recursos necessários do NLTK"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        st.warning("⚠️ Erro ao baixar recursos do NLTK. Algumas funcionalidades podem não funcionar.")

download_nltk_resources()

# Configuração de estilo
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """
    Carrega e pré-processa os dados do dataset usando o caminho correto
    """
    try:
        # Usa o caminho configurado no Config
        dataset_path = Config.DATASET_PATH
        
        # Verifica se o arquivo existe
        if not os.path.exists(dataset_path):
            st.error(f"❌ Arquivo não encontrado: {dataset_path}")
            st.info("Verifique se o arquivo olist_order_reviews_dataset.csv está na pasta data/")
            return None
        
        # Carrega o dataset
        df = pd.read_csv(dataset_path)
        
        # Converte review_creation_date para datetime
        df['review_creation_date'] = pd.to_datetime(df['review_creation_date'], errors='coerce')
        
        # Remove linhas com data inválida
        df = df.dropna(subset=['review_creation_date'])
        
        # Trata valores nulos na coluna review_comment_message
        df['review_comment_message'] = df['review_comment_message'].fillna('')
        
        # Filtra apenas reviews com comentários válidos
        df = df[df['review_comment_message'].str.len() > 10]
        
        # Adiciona colunas úteis
        df['year'] = df['review_creation_date'].dt.year
        df['month'] = df['review_creation_date'].dt.month
        df['year_month'] = df['review_creation_date'].dt.to_period('M')
        
        return df
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return None

@st.cache_resource
def initialize_vector_store():
    """
    Inicializa o banco vetorial
    """
    try:
        vector_store = VectorStore()
        
        # Verifica se o índice existe
        if os.path.exists(f"{Config.FAISS_INDEX_PATH}.faiss"):
            st.info("🔄 Carregando banco vetorial existente...")
            vector_store.load_index()
        else:
            st.info("🔄 Construindo banco vetorial...")
            vector_store.build_index()
        
        return vector_store
    except Exception as e:
        st.error(f"❌ Erro ao inicializar banco vetorial: {e}")
        return None

@st.cache_resource
def initialize_sentiment_analyzer():
    """
    Inicializa o analisador de sentimentos
    """
    try:
        return SentimentAnalyzer()
    except Exception as e:
        st.error(f"❌ Erro ao inicializar analisador de sentimentos: {e}")
        return None

def clean_text(text):
    """
    Limpa e pré-processa texto para análise
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Converte para minúsculas
    text = text.lower()
    
    # Remove pontuação e números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove espaços extras
    text = ' '.join(text.split())
    
    return text

def get_stopwords():
    """
    Retorna lista de stopwords em português
    """
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('portuguese'))
    except:
        # Fallback para lista básica de stopwords em português
        stop_words = {
            'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma',
            'os', 'as', 'que', 'se', 'na', 'por', 'mais', 'as', 'como', 'mas', 'foi', 'ele',
            'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos',
            'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre',
            'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'suas', 'minha', 'têm',
            'naquele', 'essas', 'esses', 'pelos', 'elas', 'estava', 'seja', 'qual', 'será',
            'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse',
            'dele', 'tu', 'te', 'você', 'vocês', 'lhe', 'lhes', 'meu', 'minha', 'meus',
            'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas',
            'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles',
            'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive',
            'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera',
            'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos',
            'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos',
            'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja',
            'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos',
            'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria',
            'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram',
            'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam',
            'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será',
            'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos',
            'têm', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram',
            'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos',
            'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos',
            'terão', 'teria', 'teríamos', 'teriam'
        }
    
    return stop_words

def get_word_frequency(texts, stop_words, top_n=10):
    """
    Calcula a frequência de palavras em uma lista de textos
    """
    all_words = []
    
    for text in texts:
        if pd.isna(text) or text == '':
            continue
        
        # Limpa o texto
        clean_text_str = clean_text(text)
        
        # Tokeniza
        words = clean_text_str.split()
        
        # Remove stopwords
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        all_words.extend(words)
    
    # Conta frequência
    word_counts = Counter(all_words)
    
    # Retorna as top_n palavras mais frequentes
    return word_counts.most_common(top_n)

def search_similar_reviews_real(query, vector_store, top_k=5):
    """
    Busca reviews similares usando o banco vetorial real
    """
    if not query or query.strip() == '':
        return []
    
    try:
        # Busca no banco vetorial
        similar_reviews = vector_store.search_similar_reviews(query, k=top_k)
        
        # Formata os resultados
        formatted_results = []
        for review in similar_reviews:
            formatted_results.append({
                'review_score': review.get('review_score', 0),
                'review_comment_message': review.get('review_comment_message', '')[:200] + '...' if len(review.get('review_comment_message', '')) > 200 else review.get('review_comment_message', ''),
                'review_creation_date': pd.to_datetime(review.get('review_creation_date')).strftime('%Y-%m-%d') if review.get('review_creation_date') else 'N/A',
                'similarity_score': review.get('similarity_score', 0)
            })
        
        return formatted_results
    except Exception as e:
        st.error(f"❌ Erro na busca vetorial: {e}")
        return []

def main():
    """
    Função principal da aplicação
    """
    # Cabeçalho
    st.markdown('<h1 class="main-header">📊 Dashboard de Análise de Sentimento e Tópicos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Insights rápidos sobre a percepção do cliente, potencializado por busca semântica com banco vetorial real</p>', unsafe_allow_html=True)
    
    # Inicializa componentes
    with st.spinner('🔄 Inicializando componentes...'):
        # Carrega dados
        df = load_and_preprocess_data()
        
        # Inicializa banco vetorial
        vector_store = initialize_vector_store()
        
        # Inicializa analisador de sentimentos
        sentiment_analyzer = initialize_sentiment_analyzer()
    
    if df is None:
        st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo existe.")
        return
    
    if vector_store is None:
        st.warning("⚠️ Banco vetorial não disponível. Algumas funcionalidades podem não funcionar.")
    
    # Informações básicas
    st.sidebar.markdown("## 📈 Informações do Dataset")
    st.sidebar.metric("Total de Reviews", f"{len(df):,}")
    st.sidebar.metric("Período", f"{df['year'].min()} - {df['year'].max()}")
    st.sidebar.metric("Score Médio", f"{df['review_score'].mean():.2f}")
    
    if vector_store:
        st.sidebar.markdown("## 🔍 Banco Vetorial")
        st.sidebar.success("✅ Banco vetorial carregado")
        st.sidebar.metric("Reviews Indexados", f"{len(vector_store.reviews_data):,}")
    
    if sentiment_analyzer:
        st.sidebar.markdown("## 😊 Análise de Sentimentos")
        st.sidebar.success("✅ Analisador de sentimentos carregado")
    
    # 1. VISÃO GERAL DA SATISFAÇÃO
    st.markdown('<h2 class="section-header">🎯 Visão Geral da Satisfação</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Score Médio", f"{df['review_score'].mean():.2f}")
    
    with col2:
        st.metric("Score Mediano", f"{df['review_score'].median():.1f}")
    
    with col3:
        st.metric("Score Modal", f"{df['review_score'].mode().iloc[0]:.0f}")
    
    with col4:
        st.metric("Desvio Padrão", f"{df['review_score'].std():.2f}")
    
    # Distribuição das pontuações
    st.subheader("📊 Distribuição das Pontuações")
    
    score_counts = df['review_score'].value_counts().sort_index()
    score_percentages = (score_counts / len(df) * 100).round(1)
    
    # Gráfico de barras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Contagem absoluta
    ax1.bar(score_counts.index, score_counts.values, color='skyblue', alpha=0.7)
    ax1.set_title('Contagem de Reviews por Score')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Quantidade de Reviews')
    ax1.grid(True, alpha=0.3)
    
    # Porcentagem
    ax2.bar(score_percentages.index, score_percentages.values, color='lightcoral', alpha=0.7)
    ax2.set_title('Distribuição Percentual por Score')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Porcentagem (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interpretação
    st.info("""
    **💡 Interpretação:** 
    - A maioria dos clientes está satisfeita (scores 4-5)
    - Scores baixos (1-2) indicam problemas que precisam de atenção
    - A distribuição mostra a qualidade geral do serviço
    """)
    
    # 2. TENDÊNCIA DE SATISFAÇÃO AO LONGO DO TEMPO
    st.markdown('<h2 class="section-header">📈 Tendência de Satisfação ao Longo do Tempo</h2>', unsafe_allow_html=True)
    
    # Filtro de ano
    years = sorted(df['year'].unique())
    selected_year = st.selectbox("Selecione o ano para análise:", years, index=len(years)-1)
    
    # Filtra dados por ano
    df_year = df[df['year'] == selected_year]
    
    if len(df_year) > 0:
        # Calcula média mensal
        monthly_avg = df_year.groupby('month')['review_score'].agg(['mean', 'count']).reset_index()
        monthly_avg.columns = ['Mês', 'Score Médio', 'Quantidade de Reviews']
        
        # Gráfico de linha
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(monthly_avg['Mês'], monthly_avg['Score Médio'], marker='o', linewidth=2, markersize=8, color='#1f77b4')
        ax.fill_between(monthly_avg['Mês'], monthly_avg['Score Médio'], alpha=0.3, color='#1f77b4')
        
        ax.set_title(f'Evolução da Satisfação Média - {selected_year}')
        ax.set_xlabel('Mês')
        ax.set_ylabel('Score Médio')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 13))
        
        # Adiciona anotações para picos e quedas
        for i, row in monthly_avg.iterrows():
            ax.annotate(f"{row['Score Médio']:.2f}", 
                       (row['Mês'], row['Score Médio']), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabela com dados
        st.subheader("📋 Dados Mensais")
        st.dataframe(monthly_avg, use_container_width=True)
        
        # Análise de tendências
        st.subheader("🔍 Análise de Tendências")
        
        # Identifica picos e quedas
        mean_score = monthly_avg['Score Médio'].mean()
        high_months = monthly_avg[monthly_avg['Score Médio'] > mean_score + 0.2]
        low_months = monthly_avg[monthly_avg['Score Médio'] < mean_score - 0.2]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(high_months) > 0:
                st.success(f"📈 **Meses com Alta Satisfação:** {', '.join(map(str, high_months['Mês'].tolist()))}")
            else:
                st.info("📊 Não foram identificados meses com satisfação excepcionalmente alta")
        
        with col2:
            if len(low_months) > 0:
                st.error(f"📉 **Meses com Baixa Satisfação:** {', '.join(map(str, low_months['Mês'].tolist()))}")
            else:
                st.info("📊 Não foram identificados meses com satisfação excepcionalmente baixa")
    
    # 3. ANÁLISE DE TÓPICOS E BUSCA SEMÂNTICA
    st.markdown('<h2 class="section-header">🔍 Análise Aprofundada dos Comentários (Simulação de Busca Vetorial)</h2>', unsafe_allow_html=True)
    
    # Obtém stopwords
    stop_words = get_stopwords()
    
    # Separa reviews positivos e negativos
    positive_reviews = df[df['review_score'].isin([4, 5])]['review_comment_message'].tolist()
    negative_reviews = df[df['review_score'].isin([1, 2])]['review_comment_message'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 O que os clientes mais elogiam (Score 4-5)")
        
        if len(positive_reviews) > 0:
            positive_words = get_word_frequency(positive_reviews, stop_words, top_n=10)
            
            # Gráfico de palavras positivas
            words, counts = zip(*positive_words)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(words)), counts, color='lightgreen', alpha=0.7)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Frequência')
            ax.set_title('Palavras Mais Frequentes em Reviews Positivos')
            ax.invert_yaxis()
            
            # Adiciona valores nas barras
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       str(count), ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **💡 Com Banco Vetorial:** 
            Com um banco vetorial real, poderíamos clusterizar semanticamente esses reviews 
            para identificar tópicos emergentes ou usar modelos de topic modeling mais avançados.
            """)
        else:
            st.warning("Não há reviews positivos suficientes para análise")
    
    with col2:
        st.subheader("😞 O que mais causa insatisfação (Score 1-2)")
        
        if len(negative_reviews) > 0:
            negative_words = get_word_frequency(negative_reviews, stop_words, top_n=10)
            
            # Gráfico de palavras negativas
            words, counts = zip(*negative_words)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(words)), counts, color='lightcoral', alpha=0.7)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Frequência')
            ax.set_title('Palavras Mais Frequentes em Reviews Negativos')
            ax.invert_yaxis()
            
            # Adiciona valores nas barras
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       str(count), ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **💡 Com Banco Vetorial:** 
            Similar ao anterior, com um banco vetorial, poderíamos identificar problemas raiz 
            através da similaridade semântica de frases e não apenas de palavras soltas.
            """)
        else:
            st.warning("Não há reviews negativos suficientes para análise")
    
    # 4. BUSCA DE REVIEWS SIMILARES COM BANCO VETORIAL REAL
    st.markdown('<h2 class="section-header">🔎 Busca de Reviews Similares com Banco Vetorial Real</h2>', unsafe_allow_html=True)
    
    if vector_store:
        st.success("""
        **✅ Banco Vetorial Ativo:** 
        O sistema está usando um banco vetorial FAISS real com embeddings semânticos para encontrar 
        reviews similares baseados no significado, não apenas em palavras-chave.
        """)
        
        # Campo de busca
        search_query = st.text_input(
            "Buscar Reviews Similares (Banco Vetorial Real):",
            placeholder="Digite palavras-chave para buscar reviews similares..."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            search_button = st.button("🔍 Buscar", type="primary")
        
        with col2:
            if search_button and search_query:
                with st.spinner('🔍 Buscando no banco vetorial...'):
                    similar_reviews = search_similar_reviews_real(search_query, vector_store, top_k=5)
                
                if similar_reviews:
                    st.subheader("📋 Reviews Encontrados (Banco Vetorial Real)")
                    
                    for i, review in enumerate(similar_reviews, 1):
                        with st.expander(f"Review {i} - Score: {review['review_score']} - Similaridade: {review['similarity_score']:.3f}"):
                            st.write(f"**Data:** {review['review_creation_date']}")
                            st.write(f"**Comentário:** {review['review_comment_message']}")
                            
                            # Análise de sentimento se disponível
                            if sentiment_analyzer:
                                try:
                                    sentiment = sentiment_analyzer.analyze_text_sentiment(review['review_comment_message'])
                                    st.write(f"**Análise de Sentimento:** {sentiment['sentiment']} (Confiança: {sentiment['confidence']:.2f})")
                                except:
                                    pass
                else:
                    st.warning("Nenhum review similar encontrado para esta busca.")
    else:
        st.error("❌ Banco vetorial não disponível. Não é possível realizar buscas semânticas.")
        st.info("Verifique se os arquivos do banco vetorial estão presentes na pasta data/")
    
    # 5. RESUMO E RECOMENDAÇÕES
    st.markdown('<h2 class="section-header">📋 Resumo e Recomendações</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Principais Insights")
        
        # Calcula insights
        total_reviews = len(df)
        positive_pct = len(df[df['review_score'].isin([4, 5])]) / total_reviews * 100
        negative_pct = len(df[df['review_score'].isin([1, 2])]) / total_reviews * 100
        
        st.metric("Satisfação Geral", f"{positive_pct:.1f}%", f"{positive_pct - 50:.1f}%")
        st.metric("Insatisfação", f"{negative_pct:.1f}%", f"{negative_pct - 20:.1f}%")
        
        # Recomendações baseadas nos dados
        if positive_pct > 70:
            st.success("✅ **Excelente satisfação geral!** Continue mantendo a qualidade.")
        elif positive_pct > 50:
            st.info("📊 **Satisfação moderada.** Há espaço para melhorias.")
        else:
            st.error("⚠️ **Baixa satisfação.** Ação imediata necessária.")
    
    with col2:
        st.subheader("🚀 Próximos Passos Sugeridos")
        
        recommendations = []
        
        if negative_pct > 20:
            recommendations.append("🔍 Investigar causas da insatisfação")
        
        if df['review_score'].std() > 1.5:
            recommendations.append("📊 Analisar inconsistências na qualidade")
        
        if len(negative_reviews) > 100:
            recommendations.append("🎯 Implementar melhorias baseadas em feedback negativo")
        
        if len(positive_reviews) > 100:
            recommendations.append("🌟 Reforçar pontos positivos identificados")
        
        if not recommendations:
            recommendations.append("📈 Manter monitoramento contínuo")
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Dashboard criado para análise de sentimentos da Olist</p>
        <p>💡 Potencializado por simulação de banco vetorial para busca semântica</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 