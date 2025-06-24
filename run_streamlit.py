#!/usr/bin/env python3
"""
Script para executar a aplicação Streamlit do Dashboard Olist Reviews
"""

import subprocess
import sys
import os
import time

def run_streamlit():
    """Executa a aplicação Streamlit"""
    
    print("🚀 INICIANDO DASHBOARD STREAMLIT - OLIST REVIEWS")
    print("=" * 60)
    
    # Verifica se o arquivo existe
    if not os.path.exists('app_streamlit.py'):
        print("❌ Arquivo app_streamlit.py não encontrado!")
        return False
    
    # Verifica se o dataset existe
    if not os.path.exists('data/olist_order_reviews_dataset.csv'):
        print("❌ Dataset não encontrado em data/olist_order_reviews_dataset.csv!")
        return False
    
    print("✅ Arquivos verificados com sucesso")
    print("📊 Dataset: data/olist_order_reviews_dataset.csv")
    print("🌐 Aplicação: app_streamlit.py")
    
    # Configurações do Streamlit
    port = 8501
    host = "localhost"
    
    print(f"\n🔧 Configurações:")
    print(f"   Host: {host}")
    print(f"   Porta: {port}")
    print(f"   URL: http://{host}:{port}")
    
    # Comando para executar o Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app_streamlit.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"\n📋 Comando: {' '.join(cmd)}")
    print("\n⏳ Iniciando aplicação Streamlit...")
    print("💡 Pressione Ctrl+C para parar a aplicação")
    print("=" * 60)
    
    try:
        # Executa o Streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Aplicação parada pelo usuário")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao executar Streamlit: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        return False
    
    return True

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    
    print("🔍 Verificando dependências...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Pacotes faltando: {', '.join(missing_packages)}")
        print("💡 Execute: pip install -r requirements.txt")
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True

if __name__ == "__main__":
    print("🔧 CONFIGURADOR DO DASHBOARD STREAMLIT")
    print("=" * 60)
    
    # Verifica dependências
    if not check_dependencies():
        print("\n❌ Dependências não atendidas. Instale os pacotes necessários.")
        sys.exit(1)
    
    # Executa a aplicação
    run_streamlit() 