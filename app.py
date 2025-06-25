import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Carrega o dataset
try:
    df = pd.read_csv('app/data/olist_order_reviews_dataset.csv')
    print("Dataset carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Dataset não encontrado.")
    df = pd.DataFrame()

def get_basic_stats():
    if df.empty:
        return "Dataset não carregado"
    
    stats = {
        "Total de Reviews": len(df),
        "Reviews com Comentário": df['review_comment_message'].notna().sum(),
        "Média de Avaliação": df['review_score'].mean(),
        "Mediana de Avaliação": df['review_score'].median(),
    }
    return stats

def analyze_data():
    if df.empty:
        return "❌ Dataset não carregado."
    
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

def create_score_distribution_plot():
    if df.empty:
        return None
    
    score_counts = df['review_score'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    x_values = score_counts.index.tolist()
    y_values = score_counts.values.tolist()
    bars = plt.bar(x_values, y_values, color='skyblue', alpha=0.7)
    
    for bar in bars:
        height = float(bar.get_height())
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.title('Distribuição das Avaliações', fontsize=14, fontweight='bold')
    plt.xlabel('Avaliação (1-5)', fontsize=12)
    plt.ylabel('Número de Reviews', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(1, 6))
    plt.tight_layout()
    
    return plt.gcf()

# Interface Gradio
with gr.Blocks(title="Olist Reviews - Dashboard") as demo:
    gr.Markdown("# 🚀 Olist Reviews - Análise de Dados")
    
    with gr.Tabs():
        with gr.TabItem("📊 Visão Geral"):
            analyze_btn = gr.Button("🔍 Analisar Dados", variant="primary")
            stats_output = gr.Markdown()
            analyze_btn.click(fn=analyze_data, outputs=stats_output)
        
        with gr.TabItem("📈 Gráficos"):
            dist_btn = gr.Button("📊 Distribuição de Avaliações")
            dist_plot = gr.Plot()
            dist_btn.click(fn=create_score_distribution_plot, outputs=dist_plot)
        
        with gr.TabItem("ℹ️ Informações"):
            info_text = f"""
            ## 📋 Sobre o Olist Reviews
            
            Dashboard para análise de reviews da Olist.
            
            ### 📊 Dataset
            - **Total de Reviews**: {len(df) if not df.empty else "N/A"}
            - **Período**: 2016-2018
            
            ### 🛠️ Tecnologias
            - Gradio
            - Pandas
            - Matplotlib
            """
            gr.Markdown(info_text)

# Configuração para deploy
if __name__ == "__main__":
    print("🚀 Iniciando Olist Reviews Dashboard...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 