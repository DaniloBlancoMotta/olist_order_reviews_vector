#!/usr/bin/env python3
"""
Dashboard Olist Reviews - Versão Ultra-Minimalista
Compatible com Streamlit Cloud sem dependências externas
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuração da página
st.set_page_config(
    page_title="Dashboard Olist Reviews",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS
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
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega os dados do dataset"""
    try:
        # Tenta diferentes caminhos possíveis
        possible_paths = [
            "data/olist_order_reviews_dataset.csv",
            "olist_order_reviews_dataset.csv",
            "/mount/src/olist_order_reviews_vector/data/olist_order_reviews_dataset.csv"
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            st.error("❌ Dataset não encontrado. Verifique se o arquivo CSV está disponível.")
            return None
        
        df = pd.read_csv(dataset_path)
        
        # Processamento básico
        df['review_creation_date'] = pd.to_datetime(df['review_creation_date'], errors='coerce')
        df = df.dropna(subset=['review_creation_date'])
        df['review_comment_message'] = df['review_comment_message'].fillna('')
        df = df[df['review_comment_message'].str.len() > 10]
        
        # Adiciona colunas úteis
        df['year'] = df['review_creation_date'].dt.year
        df['month'] = df['review_creation_date'].dt.month
        
        return df
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return None

def clean_text(text):
    """Limpa texto para análise"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    return text

def get_stopwords():
    """Retorna stopwords em português (hardcoded)"""
    return {
        'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma',
        'os', 'as', 'que', 'se', 'na', 'por', 'mais', 'como', 'mas', 'foi', 'ele',
        'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos',
        'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela',
        'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'suas', 'minha',
        'naquele', 'essas', 'esses', 'pelos', 'elas', 'estava', 'seja', 'qual',
        'nós', 'tenho', 'lhe', 'deles', 'pelas', 'este', 'fosse', 'dele', 'tu',
        'te', 'você', 'vocês', 'lhe', 'lhes', 'meu', 'minha', 'meus', 'minhas',
        'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas',
        'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles',
        'aquelas', 'isto', 'aquilo', 'estou', 'estamos', 'estão', 'estive',
        'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam',
        'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem',
        'estiver', 'estivermos', 'estiverem', 'hei', 'havemos', 'hão', 'houve',
        'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos',
        'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos',
        'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria',
        'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram',
        'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos',
        'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem',
        'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam',
        'tenho', 'tem', 'temos', 'têm', 'tinha', 'tínhamos', 'tinham', 'tive',
        'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos',
        'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos',
        'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos',
        'teriam', 'produto', 'entrega', 'qualidade', 'preço', 'recomendo',
        'não recomendo', 'chegou', 'veio', 'bom', 'boa', 'ruim', 'ótimo',
        'excelente', 'péssimo'
    }

def get_word_frequency(texts, stop_words, top_n=10):
    """Calcula frequência de palavras"""
    all_words = []
    
    for text in texts:
        if pd.isna(text) or text == '':
            continue
        
        clean_text_str = clean_text(text)
        words = clean_text_str.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)

def main():
    """Função principal"""
    # Cabeçalho
    st.markdown('<h1 class="main-header">📊 Dashboard Olist Reviews</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análise de sentimentos e insights dos reviews - Versão Cloud</p>', unsafe_allow_html=True)
    
    # Carrega dados
    with st.spinner('🔄 Carregando dados...'):
        df = load_data()
    
    if df is None:
        st.error("❌ Não foi possível carregar os dados.")
        st.info("💡 Verifique se o arquivo CSV está na pasta data/ ou na raiz do projeto.")
        return
    
    # Sidebar com informações
    st.sidebar.markdown("## 📈 Informações")
    st.sidebar.metric("Total de Reviews", f"{len(df):,}")
    st.sidebar.metric("Período", f"{df['year'].min()} - {df['year'].max()}")
    st.sidebar.metric("Score Médio", f"{df['review_score'].mean():.2f}")
    
    # 1. VISÃO GERAL
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
    
    # Distribuição de scores
    st.subheader("📊 Distribuição das Pontuações")
    
    score_counts = df['review_score'].value_counts().sort_index()
    
    # Gráfico Plotly
    fig = px.bar(
        x=score_counts.index, 
        y=score_counts.values,
        title="Distribuição de Scores dos Reviews",
        labels={'x': 'Score', 'y': 'Quantidade de Reviews'},
        color=score_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretação
    st.info("""
    **💡 Interpretação:** 
    - A maioria dos clientes está satisfeita (scores 4-5)
    - Scores baixos (1-2) indicam problemas que precisam de atenção
    - A distribuição mostra a qualidade geral do serviço
    """)
    
    # 2. TENDÊNCIA TEMPORAL
    st.markdown('<h2 class="section-header">📈 Tendência de Satisfação</h2>', unsafe_allow_html=True)
    
    # Filtro de ano
    years = sorted(df['year'].unique())
    selected_year = st.selectbox("Selecione o ano:", years, index=len(years)-1)
    
    df_year = df[df['year'] == selected_year]
    
    if len(df_year) > 0:
        monthly_avg = df_year.groupby('month')['review_score'].mean().reset_index()
        
        # Gráfico de linha
        fig = px.line(
            monthly_avg,
            x='month',
            y='review_score',
            title=f'Evolução da Satisfação Média - {selected_year}',
            labels={'month': 'Mês', 'review_score': 'Score Médio'},
            markers=True
        )
        
        fig.update_layout(height=400)
        fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de dados
        st.subheader("📋 Dados Mensais")
        monthly_stats = df_year.groupby('month')['review_score'].agg(['mean', 'count']).reset_index()
        monthly_stats.columns = ['Mês', 'Score Médio', 'Quantidade']
        st.dataframe(monthly_stats, use_container_width=True)
        
        # Análise de tendências
        st.subheader("🔍 Análise de Tendências")
        
        mean_score = monthly_stats['Score Médio'].mean()
        high_months = monthly_stats[monthly_stats['Score Médio'] > mean_score + 0.2]
        low_months = monthly_stats[monthly_stats['Score Médio'] < mean_score - 0.2]
        
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
    
    # 3. ANÁLISE DE TÓPICOS
    st.markdown('<h2 class="section-header">🔍 Análise de Tópicos</h2>', unsafe_allow_html=True)
    
    stop_words = get_stopwords()
    
    # Separa reviews por score
    positive_reviews = df[df['review_score'].isin([4, 5])]['review_comment_message'].tolist()
    negative_reviews = df[df['review_score'].isin([1, 2])]['review_comment_message'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Reviews Positivos (Score 4-5)")
        
        if len(positive_reviews) > 0:
            positive_words = get_word_frequency(positive_reviews, stop_words, top_n=10)
            
            if positive_words:
                words, counts = zip(*positive_words)
                
                fig = px.bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    title="Palavras Mais Frequentes em Reviews Positivos",
                    labels={'x': 'Frequência', 'y': 'Palavras'},
                    color=counts,
                    color_continuous_scale='greens'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Não há palavras suficientes para análise")
        else:
            st.warning("Não há reviews positivos suficientes")
    
    with col2:
        st.subheader("😞 Reviews Negativos (Score 1-2)")
        
        if len(negative_reviews) > 0:
            negative_words = get_word_frequency(negative_reviews, stop_words, top_n=10)
            
            if negative_words:
                words, counts = zip(*negative_words)
                
                fig = px.bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    title="Palavras Mais Frequentes em Reviews Negativos",
                    labels={'x': 'Frequência', 'y': 'Palavras'},
                    color=counts,
                    color_continuous_scale='reds'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Não há palavras suficientes para análise")
        else:
            st.warning("Não há reviews negativos suficientes")
    
    # 4. RESUMO
    st.markdown('<h2 class="section-header">📋 Resumo e Recomendações</h2>', unsafe_allow_html=True)
    
    # Calcula métricas
    total_reviews = len(df)
    positive_pct = len(df[df['review_score'].isin([4, 5])]) / total_reviews * 100
    negative_pct = len(df[df['review_score'].isin([1, 2])]) / total_reviews * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Principais Insights")
        st.metric("Satisfação Geral", f"{positive_pct:.1f}%")
        st.metric("Insatisfação", f"{negative_pct:.1f}%")
        
        if positive_pct > 70:
            st.success("✅ Excelente satisfação geral!")
        elif positive_pct > 50:
            st.info("📊 Satisfação moderada. Há espaço para melhorias.")
        else:
            st.error("⚠️ Baixa satisfação. Ação necessária.")
    
    with col2:
        st.subheader("🚀 Recomendações")
        
        recommendations = []
        
        if negative_pct > 20:
            recommendations.append("🔍 Investigar causas da insatisfação")
        
        if df['review_score'].std() > 1.5:
            recommendations.append("📊 Analisar inconsistências na qualidade")
        
        if len(negative_reviews) > 100:
            recommendations.append("🎯 Implementar melhorias baseadas em feedback")
        
        if not recommendations:
            recommendations.append("📈 Manter monitoramento contínuo")
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Dashboard Olist Reviews - Versão Cloud Minimalista</p>
        <p>🚀 Compatível com Streamlit Cloud sem dependências externas</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 