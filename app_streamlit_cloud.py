#!/usr/bin/env python3
"""
Dashboard de Análise de Sentimento - Olist Reviews
Versão otimizada para Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Dashboard Olist Reviews",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """
    Carrega e pré-processa os dados do dataset
    """
    try:
        # Carrega o dataset
        df = pd.read_csv('data/olist_order_reviews_dataset.csv')
        
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
        
        return df
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
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

def simulate_vector_search(query, df, top_k=5):
    """
    Simula busca em banco vetorial
    """
    if not query or query.strip() == '':
        return []
    
    # Simulação simples: busca por palavras-chave
    query_lower = query.lower()
    
    # Filtra reviews que contêm palavras da query
    matching_reviews = []
    
    for _, row in df.iterrows():
        review_text = str(row['review_comment_message']).lower()
        if any(word in review_text for word in query_lower.split()):
            matching_reviews.append({
                'review_score': row['review_score'],
                'review_comment_message': row['review_comment_message'][:200] + '...' if len(row['review_comment_message']) > 200 else row['review_comment_message'],
                'review_creation_date': row['review_creation_date'].strftime('%Y-%m-%d'),
                'similarity_score': np.random.uniform(0.7, 1.0)
            })
    
    # Ordena por score de similaridade e retorna top_k
    matching_reviews.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matching_reviews[:top_k]

def main():
    """
    Função principal da aplicação
    """
    # Cabeçalho
    st.markdown('<h1 class="main-header">📊 Dashboard de Análise de Sentimento</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Insights rápidos sobre a percepção do cliente</p>', unsafe_allow_html=True)
    
    # Carrega dados
    with st.spinner('🔄 Carregando dados...'):
        df = load_and_preprocess_data()
    
    if df is None:
        st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo existe.")
        return
    
    # Informações básicas
    st.sidebar.markdown("## 📈 Informações do Dataset")
    st.sidebar.metric("Total de Reviews", f"{len(df):,}")
    st.sidebar.metric("Período", f"{df['year'].min()} - {df['year'].max()}")
    st.sidebar.metric("Score Médio", f"{df['review_score'].mean():.2f}")
    
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
    
    # Tabela de distribuição
    dist_data = pd.DataFrame({
        'Score': score_counts.index,
        'Quantidade': score_counts.values,
        'Porcentagem (%)': score_percentages.values
    })
    
    st.dataframe(dist_data, use_container_width=True)
    
    # Gráfico usando streamlit nativo
    st.subheader("Contagem de Reviews por Score")
    st.bar_chart(score_counts)
    
    st.subheader("Distribuição Percentual por Score")
    st.bar_chart(score_percentages)
    
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
        
        # Tabela com dados
        st.subheader("📋 Dados Mensais")
        st.dataframe(monthly_avg, use_container_width=True)
        
        # Gráfico de linha usando streamlit
        st.subheader(f"Evolução da Satisfação Média - {selected_year}")
        chart_data = pd.DataFrame({
            'Mês': monthly_avg['Mês'],
            'Score Médio': monthly_avg['Score Médio']
        }).set_index('Mês')
        st.line_chart(chart_data)
        
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
    
    # 3. ANÁLISE DE TÓPICOS
    st.markdown('<h2 class="section-header">🔍 Análise dos Comentários</h2>', unsafe_allow_html=True)
    
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
            
            # Tabela de palavras positivas
            pos_df = pd.DataFrame(positive_words, columns=['Palavra', 'Frequência'])
            st.dataframe(pos_df, use_container_width=True)
            
            st.info("""
            **💡 Insights:** 
            Palavras mais frequentes em reviews positivos mostram o que está funcionando bem.
            """)
        else:
            st.warning("Não há reviews positivos suficientes para análise")
    
    with col2:
        st.subheader("😞 O que mais causa insatisfação (Score 1-2)")
        
        if len(negative_reviews) > 0:
            negative_words = get_word_frequency(negative_reviews, stop_words, top_n=10)
            
            # Tabela de palavras negativas
            neg_df = pd.DataFrame(negative_words, columns=['Palavra', 'Frequência'])
            st.dataframe(neg_df, use_container_width=True)
            
            st.info("""
            **💡 Insights:** 
            Palavras mais frequentes em reviews negativos mostram problemas que precisam ser resolvidos.
            """)
        else:
            st.warning("Não há reviews negativos suficientes para análise")
    
    # 4. SIMULAÇÃO DE BUSCA
    st.markdown('<h2 class="section-header">🔎 Simulação de Busca de Reviews</h2>', unsafe_allow_html=True)
    
    st.info("""
    **💡 Como Funciona a Busca:**
    Simulação de busca por palavras-chave nos comentários dos clientes.
    """)
    
    # Campo de busca
    search_query = st.text_input(
        "Buscar Reviews Similares:",
        placeholder="Digite palavras-chave para buscar reviews similares..."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        search_button = st.button("🔍 Buscar", type="primary")
    
    with col2:
        if search_button and search_query:
            st.write(f"**Buscando reviews similares a:** '{search_query}'")
            
            # Simula busca
            similar_reviews = simulate_vector_search(search_query, df, top_k=5)
            
            if similar_reviews:
                st.subheader("📋 Reviews Encontrados")
                
                for i, review in enumerate(similar_reviews, 1):
                    with st.expander(f"Review {i} - Score: {review['review_score']} - Similaridade: {review['similarity_score']:.3f}"):
                        st.write(f"**Data:** {review['review_creation_date']}")
                        st.write(f"**Comentário:** {review['review_comment_message']}")
            else:
                st.warning("Nenhum review similar encontrado para esta busca.")
    
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
        <p>💡 Versão otimizada para Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 