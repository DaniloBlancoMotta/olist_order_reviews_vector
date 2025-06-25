#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Exploratória dos Dados - Reviews Olist
Visualizações gráficas completas do dataset de reviews
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
from datetime import datetime
import re
from collections import Counter

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore')

# Configuração para exibir todas as colunas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def carregar_dados():
    """Carrega e prepara os dados"""
    print("Carregando dados...")
    df_reviews = pd.read_csv('../app/data/olist_order_reviews_dataset.csv')
    
    print(f"Shape do dataset: {df_reviews.shape}")
    print(f"Período dos dados: {df_reviews['review_creation_date'].min()} a {df_reviews['review_creation_date'].max()}")
    
    # Limpeza e preparação dos dados
    df_reviews['review_creation_date'] = pd.to_datetime(df_reviews['review_creation_date'])
    df_reviews['review_answer_timestamp'] = pd.to_datetime(df_reviews['review_answer_timestamp'])
    
    # Tratar valores nulos nos comentários
    df_reviews['review_comment_message'] = df_reviews['review_comment_message'].fillna('')
    
    # Adicionar colunas úteis
    df_reviews['comment_length'] = df_reviews['review_comment_message'].str.len()
    df_reviews['word_count'] = df_reviews['review_comment_message'].str.split().str.len()
    df_reviews['has_comment'] = df_reviews['review_comment_message'].str.len() > 0
    
    return df_reviews

def analise_distribuicao_notas(df_reviews):
    """Análise da distribuição das notas"""
    print("\n=== ANÁLISE DA DISTRIBUIÇÃO DAS NOTAS ===")
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Histograma das notas
    sns.histplot(data=df_reviews, x='review_score', bins=5, ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Distribuição das Notas dos Reviews', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Nota (1-5)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for i in range(1, 6):
        count = len(df_reviews[df_reviews['review_score'] == i])
        percentage = (count / len(df_reviews)) * 100
        ax1.text(i, count + 1000, f'{count:,}\n({percentage:.1f}%)', 
                 ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Countplot das notas
    sns.countplot(data=df_reviews, x='review_score', ax=ax2, palette='viridis')
    ax2.set_title('Contagem de Reviews por Nota', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Nota (1-5)', fontsize=12)
    ax2.set_ylabel('Quantidade de Reviews', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for i in range(5):
        count = ax2.patches[i].get_height()
        ax2.text(ax2.patches[i].get_x() + ax2.patches[i].get_width()/2, count + 500, 
                 f'{int(count):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Análise das notas mais comuns
    score_counts = df_reviews['review_score'].value_counts().sort_index()
    for score, count in score_counts.items():
        percentage = (count / len(df_reviews)) * 100
        print(f"Nota {score}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\nNota mais comum: {score_counts.idxmax()} ({score_counts.max():,} reviews)")
    print(f"Nota menos comum: {score_counts.idxmin()} ({score_counts.min():,} reviews)")
    print(f"Média das notas: {df_reviews['review_score'].mean():.2f}")

def analise_temporal(df_reviews):
    """Análise temporal dos reviews"""
    print("\n=== ANÁLISE TEMPORAL DOS REVIEWS ===")
    
    # Preparar dados temporais
    df_reviews['date'] = df_reviews['review_creation_date'].dt.date
    df_reviews['month'] = df_reviews['review_creation_date'].dt.to_period('M')
    df_reviews['week'] = df_reviews['review_creation_date'].dt.to_period('W')
    
    # Agregações por diferentes períodos
    daily_reviews = df_reviews.groupby('date').size().reset_index(name='count')
    weekly_reviews = df_reviews.groupby('week').size().reset_index(name='count')
    monthly_reviews = df_reviews.groupby('month').size().reset_index(name='count')
    
    # Converter períodos para datetime para plotagem
    weekly_reviews['week_dt'] = weekly_reviews['week'].dt.to_timestamp()
    monthly_reviews['month_dt'] = monthly_reviews['month'].dt.to_timestamp()
    
    # Criar figura com subplots para diferentes granularidades temporais
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))
    
    # Gráfico 1: Tendência diária
    ax1.plot(daily_reviews['date'], daily_reviews['count'], linewidth=1, alpha=0.7, color='blue')
    ax1.set_title('Volume de Reviews por Dia', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data', fontsize=12)
    ax1.set_ylabel('Número de Reviews', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Tendência semanal
    ax2.plot(weekly_reviews['week_dt'], weekly_reviews['count'], linewidth=2, color='green', marker='o', markersize=4)
    ax2.set_title('Volume de Reviews por Semana', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Semana', fontsize=12)
    ax2.set_ylabel('Número de Reviews', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Gráfico 3: Tendência mensal
    ax3.plot(monthly_reviews['month_dt'], monthly_reviews['count'], linewidth=3, color='red', marker='s', markersize=8)
    ax3.set_title('Volume de Reviews por Mês', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mês', fontsize=12)
    ax3.set_ylabel('Número de Reviews', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Análise sazonal por mês
    monthly_avg = df_reviews.groupby(df_reviews['review_creation_date'].dt.month)['review_score'].agg(['count', 'mean']).reset_index()
    monthly_avg.columns = ['month', 'review_count', 'avg_score']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Volume por mês
    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    ax1.bar(months, monthly_avg['review_count'], color='lightcoral', alpha=0.7)
    ax1.set_title('Volume de Reviews por Mês do Ano', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mês', fontsize=12)
    ax1.set_ylabel('Número de Reviews', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Média das notas por mês
    ax2.bar(months, monthly_avg['avg_score'], color='lightblue', alpha=0.7)
    ax2.set_title('Média das Notas por Mês do Ano', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mês', fontsize=12)
    ax2.set_ylabel('Média das Notas', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Análise dos picos e quedas
    print(f"Período total: {df_reviews['review_creation_date'].min().strftime('%d/%m/%Y')} a {df_reviews['review_creation_date'].max().strftime('%d/%m/%Y')}")
    print(f"Total de dias com reviews: {len(daily_reviews)}")
    print(f"Média diária de reviews: {daily_reviews['count'].mean():.1f}")
    print(f"Dia com mais reviews: {daily_reviews.loc[daily_reviews['count'].idxmax(), 'date']} ({daily_reviews['count'].max()} reviews)")
    print(f"Dia com menos reviews: {daily_reviews.loc[daily_reviews['count'].idxmin(), 'date']} ({daily_reviews['count'].min()} reviews)")

def analise_comentarios(df_reviews):
    """Análise dos comentários"""
    print("\n=== ANÁLISE DOS COMENTÁRIOS ===")
    
    # Estatísticas dos comentários
    print(f"Reviews com comentários: {df_reviews['has_comment'].sum():,} ({df_reviews['has_comment'].mean()*100:.1f}%)")
    print(f"Reviews sem comentários: {(~df_reviews['has_comment']).sum():,} ({(~df_reviews['has_comment']).mean()*100:.1f}%)")
    
    df_with_comments = df_reviews[df_reviews['has_comment']]
    print(f"\nComentários com texto:")
    print(f"  - Média de caracteres: {df_with_comments['comment_length'].mean():.1f}")
    print(f"  - Mediana de caracteres: {df_with_comments['comment_length'].median():.1f}")
    print(f"  - Média de palavras: {df_with_comments['word_count'].mean():.1f}")
    print(f"  - Mediana de palavras: {df_with_comments['word_count'].median():.1f}")
    
    # Criar figura com subplots para análise dos comentários
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Distribuição do tamanho dos comentários (caracteres)
    sns.histplot(data=df_with_comments, x='comment_length', bins=50, ax=ax1, color='lightgreen', alpha=0.7)
    ax1.set_title('Distribuição do Tamanho dos Comentários (Caracteres)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Número de Caracteres', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Distribuição do número de palavras
    sns.histplot(data=df_with_comments, x='word_count', bins=30, ax=ax2, color='lightcoral', alpha=0.7)
    ax2.set_title('Distribuição do Número de Palavras nos Comentários', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Número de Palavras', fontsize=12)
    ax2.set_ylabel('Frequência', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Tamanho do comentário vs Nota
    sns.boxplot(data=df_with_comments, x='review_score', y='comment_length', ax=ax3, palette='Set3')
    ax3.set_title('Tamanho dos Comentários por Nota', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Nota', fontsize=12)
    ax3.set_ylabel('Número de Caracteres', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Proporção de reviews com comentários por nota
    comment_proportion = df_reviews.groupby('review_score')['has_comment'].mean().reset_index()
    sns.barplot(data=comment_proportion, x='review_score', y='has_comment', ax=ax4, palette='viridis')
    ax4.set_title('Proporção de Reviews com Comentários por Nota', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Nota', fontsize=12)
    ax4.set_ylabel('Proporção com Comentários', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def nuvem_palavras(df_reviews):
    """Cria nuvem de palavras dos comentários"""
    print("\n=== NUVEM DE PALAVRAS ===")
    
    def clean_text(text):
        """Limpa o texto para a nuvem de palavras"""
        if pd.isna(text) or text == '':
            return ''
        # Converter para minúsculas e remover caracteres especiais
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    
    df_with_comments = df_reviews[df_reviews['has_comment']]
    
    # Preparar texto para nuvem de palavras
    all_comments = ' '.join(df_with_comments['review_comment_message'].apply(clean_text))
    
    # Criar nuvem de palavras
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis',
        random_state=42
    ).generate(all_comments)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras dos Comentários dos Reviews', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Análise das palavras mais frequentes
    def extract_words(text):
        """Extrai palavras do texto"""
        if pd.isna(text) or text == '':
            return []
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        # Remover palavras comuns em português
        stop_words = {'que', 'com', 'para', 'uma', 'por', 'mais', 'como', 'mas', 'foi', 'ele', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'suas', 'minha', 'têm', 'naquele', 'essas', 'esses', 'pelos', 'elas', 'estava', 'seja', 'qual', 'nossa', 'nossos', 'nossa', 'nossas', 'ou', 'onde', 'meu', 'minhas', 'numa', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa', 'vocês', 'já', 'ou', 'um', 'após', 'até', 'sem', 'sob', 'sobre', 'entre', 'contra', 'desde', 'durante', 'para', 'perante', 'segundo', 'conforme', 'consoante', 'mediante', 'salvo', 'tirante', 'visto'}
        return [word for word in words if word not in stop_words]
    
    all_words = []
    for comment in df_with_comments['review_comment_message']:
        all_words.extend(extract_words(comment))
    
    word_freq = Counter(all_words)
    most_common_words = word_freq.most_common(20)
    
    print("PALAVRAS MAIS FREQUENTES NOS COMENTÁRIOS:")
    for word, count in most_common_words:
        print(f"{word}: {count} vezes")

def analise_por_nota(df_reviews):
    """Análise detalhada por nota"""
    print("\n=== ANÁLISE POR NOTA ===")
    
    def clean_text(text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    
    # Análise detalhada por nota
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, score in enumerate([1, 2, 3, 4, 5]):
        score_data = df_reviews[df_reviews['review_score'] == score]
        score_comments = score_data[score_data['has_comment']]
        
        if len(score_comments) > 0:
            # Nuvem de palavras para cada nota
            text = ' '.join(score_comments['review_comment_message'].apply(clean_text))
            if text.strip():
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=50,
                    colormap='viridis',
                    random_state=42
                ).generate(text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'Nota {score} - Nuvem de Palavras', fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'Sem comentários\npara nota {score}', 
                            ha='center', va='center', transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'Nota {score}', fontweight='bold')
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'Sem comentários\npara nota {score}', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=14)
            axes[i].set_title(f'Nota {score}', fontweight='bold')
            axes[i].axis('off')
    
    # Remover o último subplot se não for usado
    axes[5].remove()
    
    plt.tight_layout()
    plt.show()

def main():
    """Função principal que executa todas as análises"""
    print("🚀 INICIANDO ANÁLISE EXPLORATÓRIA DOS REVIEWS OLIST")
    print("=" * 60)
    
    # Carregar dados
    df_reviews = carregar_dados()
    
    # Executar análises
    analise_distribuicao_notas(df_reviews)
    analise_temporal(df_reviews)
    analise_comentarios(df_reviews)
    nuvem_palavras(df_reviews)
    analise_por_nota(df_reviews)
    
    print("\n" + "=" * 60)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("\n📊 RESUMO DAS VISUALIZAÇÕES CRIADAS:")
    print("1. Distribuição das notas (histograma e countplot)")
    print("2. Tendência temporal (diária, semanal e mensal)")
    print("3. Análise sazonal por mês")
    print("4. Características dos comentários")
    print("5. Nuvem de palavras geral")
    print("6. Nuvens de palavras por nota")
    print("\n💡 PRINCIPAIS INSIGHTS:")
    print("- Distribuição das notas e tendências")
    print("- Padrões temporais e sazonalidade")
    print("- Características dos comentários dos clientes")
    print("- Palavras mais frequentes por nível de satisfação")

if __name__ == "__main__":
    main() 