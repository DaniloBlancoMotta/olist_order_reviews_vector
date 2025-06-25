import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# ------------------------------------------------------------------
# 1) Load CSV data once

# Adaptação: Carrega o dataset de reviews do Olist
try:
    df = pd.read_csv('app/data/olist_order_reviews_dataset.csv')
    print("Dataset carregado com sucesso!")
    print(f"Número de linhas: {len(df)}")
    print("Colunas:", df.columns.tolist())
except FileNotFoundError:
    print("Erro: O arquivo 'olist_order_reviews_dataset.csv' não foi encontrado.")
    print("Por favor, verifique se o arquivo está no diretório correto.")
    df = pd.DataFrame()  # Cria um DataFrame vazio para evitar erros posteriores

# ------------------------------------------------------------------
# 2) Funções de análise de dados

def get_basic_stats():
    """Estatísticas básicas do dataset"""
    if df.empty:
        return "Dataset não carregado"
    
    stats = {
        "Total de Reviews": len(df),
        "Reviews com Comentário": df['review_comment_message'].notna().sum(),
        "Reviews sem Comentário": df['review_comment_message'].isna().sum(),
        "Média de Avaliação": df['review_score'].mean(),
        "Mediana de Avaliação": df['review_score'].median(),
        "Desvio Padrão": df['review_score'].std()
    }
    
    return stats

def get_score_distribution():
    """Distribuição das avaliações"""
    if df.empty:
        return None
    
    score_counts = df['review_score'].value_counts().sort_index()
    return score_counts

def get_sentiment_analysis():
    """Análise básica de sentimentos baseada na pontuação"""
    if df.empty:
        return None
    
    # Classificação simples baseada na pontuação
    def classify_sentiment(score):
        if score >= 4:
            return 'Positivo'
        elif score <= 2:
            return 'Negativo'
        else:
            return 'Neutro'
    
    df_copy = df.copy()
    df_copy['sentiment'] = df_copy['review_score'].apply(classify_sentiment)
    sentiment_counts = df_copy['sentiment'].value_counts()
    
    return sentiment_counts

# ------------------------------------------------------------------
# 3) Funções para criação de gráficos

def create_score_distribution_plot():
    """Cria gráfico de distribuição das avaliações"""
    score_counts = get_score_distribution()
    if score_counts is None:
        return None
    
    plt.figure(figsize=(10, 6))
    x_values = score_counts.index.tolist()  # Converte para lista
    y_values = score_counts.values.tolist()  # Converte para lista
    bars = plt.bar(x_values, y_values, color='skyblue', alpha=0.7)
    
    # Adiciona valores nas barras
    for bar in bars:
        height = float(bar.get_height())
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.title('Distribuição das Avaliações dos Produtos', fontsize=14, fontweight='bold')
    plt.xlabel('Avaliação (1-5)', fontsize=12)
    plt.ylabel('Número de Reviews', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(1, 6))
    plt.tight_layout()
    
    return plt.gcf()

def create_sentiment_pie_chart():
    """Cria gráfico de pizza para sentimentos"""
    sentiment_counts = get_sentiment_analysis()
    if sentiment_counts is None:
        return None
    
    plt.figure(figsize=(8, 8))
    colors = ['#2E8B57', '#FF6B6B', '#FFD93D']  # Verde, Vermelho, Amarelo
    
    labels = [str(label) for label in sentiment_counts.index]
    values = sentiment_counts.values.tolist()  # Converte para lista
    plt.pie(values, labels=labels, 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Distribuição de Sentimentos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return plt.gcf()

def create_monthly_trend():
    """Cria gráfico de tendência mensal"""
    if df.empty:
        return None
    
    try:
        # Converte review_creation_date para datetime
        df_copy = df.copy()
        df_copy['review_creation_date'] = pd.to_datetime(df_copy['review_creation_date'])
        
        # Agrupa por mês
        monthly_data = df_copy.groupby(df_copy['review_creation_date'].dt.to_period('M')).agg({
            'review_score': ['count', 'mean']
        }).round(2)
        
        monthly_data.columns = ['Total_Reviews', 'Avaliação_Média']
        
        plt.figure(figsize=(12, 6))
        
        # Gráfico de linha para total de reviews
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(range(len(monthly_data)), monthly_data['Total_Reviews'], 
                         color='blue', marker='o', label='Total de Reviews')
        line2 = ax2.plot(range(len(monthly_data)), monthly_data['Avaliação_Média'], 
                         color='red', marker='s', label='Avaliação Média')
        
        ax1.set_xlabel('Mês', fontsize=12)
        ax1.set_ylabel('Total de Reviews', color='blue', fontsize=12)
        ax2.set_ylabel('Avaliação Média', color='red', fontsize=12)
        
        plt.title('Tendência Mensal de Reviews', fontsize=14, fontweight='bold')
        plt.xticks(range(len(monthly_data)), [str(x) for x in monthly_data.index], rotation=45)
        
        # Legenda
        lines = line1 + line2
        labels = [str(l.get_label()) for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    except Exception as e:
        print(f"Erro ao criar gráfico de tendência: {e}")
        return None

# ------------------------------------------------------------------
# 4) Interface Gradio

def analyze_data():
    """Função principal de análise"""
    if df.empty:
        return "❌ Dataset não carregado. Verifique se o arquivo CSV existe."
    
    # Estatísticas básicas
    stats = get_basic_stats()
    if isinstance(stats, str):
        return stats
    
    stats_text = "📊 **ESTATÍSTICAS BÁSICAS**\n\n"
    for key, value in stats.items():
        if isinstance(value, float):
            stats_text += f"**{key}:** {value:.2f}\n"
        else:
            stats_text += f"**{key}:** {value:,}\n"
    
    return stats_text

def search_reviews(product_id):
    """Busca reviews por ID do produto"""
    if df.empty:
        return "❌ Dataset não carregado."
    
    if not product_id:
        return "⚠️ Por favor, insira um ID de produto."
    
    # Busca reviews do produto
    product_reviews = df[df['product_id'] == product_id]
    
    if product_reviews.empty:
        return f"❌ Nenhum review encontrado para o produto: {product_id}"
    
    # Estatísticas do produto
    avg_score = product_reviews['review_score'].mean()
    total_reviews = len(product_reviews)
    reviews_with_comments = product_reviews['review_comment_message'].notna().sum()
    
    result = f"📦 **PRODUTO: {product_id}**\n\n"
    result += f"**Avaliação Média:** {avg_score:.2f}/5\n"
    result += f"**Total de Reviews:** {total_reviews}\n"
    result += f"**Reviews com Comentário:** {reviews_with_comments}\n\n"
    
    # Mostra alguns reviews recentes
    recent_reviews = product_reviews.head(5)
    result += "📝 **REVIEWS RECENTES:**\n\n"
    
    for idx, row in recent_reviews.iterrows():
        score = row['review_score']
        comment = row['review_comment_message'] if pd.notna(row['review_comment_message']) else "Sem comentário"
        date = row['review_creation_date']
        
        result += f"**{score}⭐** - {date}\n"
        result += f"{comment[:100]}{'...' if len(comment) > 100 else ''}\n\n"
    
    return result

# ------------------------------------------------------------------
# 5) Interface Gradio

def create_interface():
    """Cria a interface Gradio"""
    
    with gr.Blocks(title="Olist Reviews - Análise de Dados") as demo:
        gr.Markdown("# 🚀 Olist Reviews - Dashboard de Análise")
        gr.Markdown("### Sistema de Análise Inteligente de Avaliações")
        
        with gr.Tabs():
            # Tab 1: Visão Geral
            with gr.TabItem("📊 Visão Geral"):
                gr.Markdown("### Análise Geral dos Dados")
                
                analyze_btn = gr.Button("🔍 Analisar Dados", variant="primary")
                stats_output = gr.Markdown()
                
                analyze_btn.click(
                    fn=analyze_data,
                    outputs=stats_output
                )
            
            # Tab 2: Gráficos
            with gr.TabItem("📈 Gráficos"):
                gr.Markdown("### Visualizações dos Dados")
                
                with gr.Row():
                    with gr.Column():
                        dist_btn = gr.Button("📊 Distribuição de Avaliações")
                        dist_plot = gr.Plot()
                    
                    with gr.Column():
                        sentiment_btn = gr.Button("😊 Análise de Sentimentos")
                        sentiment_plot = gr.Plot()
                
                with gr.Row():
                    trend_btn = gr.Button("📈 Tendência Mensal")
                    trend_plot = gr.Plot()
                
                dist_btn.click(
                    fn=create_score_distribution_plot,
                    outputs=dist_plot
                )
                
                sentiment_btn.click(
                    fn=create_sentiment_pie_chart,
                    outputs=sentiment_plot
                )
                
                trend_btn.click(
                    fn=create_monthly_trend,
                    outputs=trend_plot
                )
            
            # Tab 3: Busca de Produtos
            with gr.TabItem("🔍 Buscar Produtos"):
                gr.Markdown("### Busca de Reviews por Produto")
                
                with gr.Row():
                    product_input = gr.Textbox(
                        label="ID do Produto",
                        placeholder="Digite o ID do produto...",
                        info="Ex: 1e9e8ef04dacdef4f5d6919c"
                    )
                    search_btn = gr.Button("🔍 Buscar", variant="primary")
                
                search_output = gr.Markdown()
                
                search_btn.click(
                    fn=search_reviews,
                    inputs=product_input,
                    outputs=search_output
                )
            
            # Tab 4: Informações
            with gr.TabItem("ℹ️ Informações"):
                gr.Markdown("### Sobre o Sistema")
                
                info_text = f"""
                ## 📋 Sobre o Olist Reviews
                
                Este dashboard permite analisar os dados de reviews da Olist, incluindo:
                
                - **📊 Estatísticas Gerais**: Visão geral dos dados
                - **📈 Visualizações**: Gráficos e análises visuais
                - **🔍 Busca de Produtos**: Análise específica por produto
                - **😊 Análise de Sentimentos**: Classificação automática
                
                ### 🛠️ Tecnologias Utilizadas
                - **Gradio**: Interface web interativa
                - **Pandas**: Manipulação de dados
                - **Matplotlib**: Visualizações
                
                ### 📊 Dataset
                - **Arquivo**: olist_order_reviews_dataset.csv
                - **Total de Reviews**: {len(df) if not df.empty else "N/A"}
                - **Período**: 2016-2018
                
                ### 🚀 Funcionalidades
                1. **Análise Estatística**: Métricas e indicadores
                2. **Visualizações**: Gráficos interativos
                3. **Busca Inteligente**: Filtros por produto
                4. **Interface Responsiva**: Fácil de usar
                """
                
                gr.Markdown(info_text)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("### 🎯 Sistema Olist Reviews - Análise Inteligente de Avaliações")
        gr.Markdown("*Desenvolvido com Gradio e Python*")
    
    return demo

# ------------------------------------------------------------------
# 6) Execução da aplicação

if __name__ == "__main__":
    print("🚀 Iniciando Olist Reviews Dashboard...")
    print(f"📊 Dataset carregado: {len(df)} reviews")
    
    # Cria e executa a interface
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,  # Mudou para porta 7862
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True  # Abre o navegador automaticamente
    ) 