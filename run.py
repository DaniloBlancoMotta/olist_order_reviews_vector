#!/usr/bin/env python3
"""
Script principal para executar a aplicação de análise de sentimentos
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import sentence_transformers
        import faiss
        import transformers
        print("✅ Todas as dependências estão instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependência não encontrada: {e}")
        print("Execute: pip install -r requirements.txt")
        return False

def check_dataset():
    """Verifica se o dataset existe"""
    dataset_path = "data/olist_order_reviews_dataset.csv"
    if Path(dataset_path).exists():
        print(f"✅ Dataset encontrado: {dataset_path}")
        return True
    else:
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return False

def build_index():
    """Constrói o índice FAISS"""
    print("🔨 Construindo índice FAISS...")
    try:
        result = subprocess.run([sys.executable, "scripts/build_index.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Índice construído com sucesso!")
            return True
        else:
            print(f"❌ Erro ao construir índice: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erro ao executar script: {e}")
        return False

def start_api():
    """Inicia a API FastAPI"""
    print("🚀 Iniciando API...")
    print("📖 Documentação disponível em: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("⏹️  Pressione Ctrl+C para parar")
    
    try:
        subprocess.run([sys.executable, "src/olist_reviews/api/main.py"])
    except KeyboardInterrupt:
        print("\n👋 API encerrada")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="API de Análise de Sentimentos - Olist Reviews")
    parser.add_argument("--build-index", action="store_true", 
                       help="Constrói o índice FAISS")
    parser.add_argument("--start-api", action="store_true", 
                       help="Inicia a API FastAPI")
    parser.add_argument("--full-setup", action="store_true", 
                       help="Executa setup completo (build-index + start-api)")
    
    args = parser.parse_args()
    
    print("🚀 API de Análise de Sentimentos - Olist Reviews")
    print("=" * 50)
    
    # Verificações iniciais
    if not check_dependencies():
        sys.exit(1)
    
    if not check_dataset():
        sys.exit(1)
    
    # Execução baseada nos argumentos
    if args.build_index:
        if not build_index():
            sys.exit(1)
    
    elif args.start_api:
        start_api()
    
    elif args.full_setup:
        print("🔧 Executando setup completo...")
        if not build_index():
            sys.exit(1)
        start_api()
    
    else:
        # Modo interativo
        print("\nEscolha uma opção:")
        print("1. Construir índice FAISS")
        print("2. Iniciar API")
        print("3. Setup completo")
        print("4. Sair")
        
        while True:
            try:
                choice = input("\nDigite sua escolha (1-4): ").strip()
                
                if choice == "1":
                    if not build_index():
                        break
                elif choice == "2":
                    start_api()
                    break
                elif choice == "3":
                    if not build_index():
                        break
                    start_api()
                    break
                elif choice == "4":
                    print("👋 Até logo!")
                    break
                else:
                    print("❌ Opção inválida. Digite 1, 2, 3 ou 4.")
                    
            except KeyboardInterrupt:
                print("\n👋 Até logo!")
                break

if __name__ == "__main__":
    main() 