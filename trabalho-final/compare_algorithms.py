#!/usr/bin/env python3
"""
Script para comparar os resultados dos algoritmos sequencial e paralelo de seleção de instâncias.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rf_modules.logger_setup import setup_logger

# Configuração
logger = setup_logger()
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_latest_metrics():
    """Carrega as métricas mais recentes dos algoritmos"""
    sequential_dir = Path("results/instance_selection")
    parallel_dir = Path("results/parallel_instance_selection")
    
    # Busca arquivos de métricas mais recentes
    sequential_files = list(sequential_dir.glob("sequential_performance_metrics_*.csv"))
    parallel_files = list(parallel_dir.glob("parallel_performance_metrics_*.csv"))
    
    if not sequential_files or not parallel_files:
        logger.error("Arquivos de métricas não encontrados. Execute os scripts de seleção primeiro.")
        return None, None
    
    # Pega os arquivos mais recentes
    latest_sequential = max(sequential_files, key=lambda x: x.stat().st_mtime)
    latest_parallel = max(parallel_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Carregando métricas sequenciais: {latest_sequential}")
    logger.info(f"Carregando métricas paralelas: {latest_parallel}")
    
    df_sequential = pd.read_csv(latest_sequential)
    df_parallel = pd.read_csv(latest_parallel)
    
    return df_sequential, df_parallel

def compare_algorithms(df_sequential, df_parallel):
    """Compara os algoritmos e gera relatório"""
    
    # Combina os dados
    df_combined = pd.concat([df_sequential, df_parallel], ignore_index=True)
    
    # Estatísticas descritivas
    logger.info("=== COMPARAÇÃO DE ALGORITMOS ===")
    
    # Comparação de tempo de execução
    seq_time = df_sequential['execution_time'].mean()
    par_time = df_parallel['execution_time'].mean()
    speedup = seq_time / par_time if par_time > 0 else 0
    
    logger.info(f"Tempo médio sequencial: {seq_time:.2f}s")
    logger.info(f"Tempo médio paralelo: {par_time:.2f}s")
    logger.info(f"Speedup real: {speedup:.2f}x")
    
    # Comparação de taxa de redução
    seq_reduction = df_sequential['reduction_rate'].mean()
    par_reduction = df_parallel['reduction_rate'].mean()
    
    logger.info(f"Taxa de redução média sequencial: {seq_reduction:.2f}%")
    logger.info(f"Taxa de redução média paralela: {par_reduction:.2f}%")
    
    # Comparação de fitness (se disponível)
    if 'best_fitness' in df_sequential.columns:
        seq_fitness = df_sequential['best_fitness'].mean()
        logger.info(f"Fitness médio sequencial: {seq_fitness:.4f}")
    
    # Eficiência paralela
    if 'num_workers' in df_parallel.columns:
        workers = df_parallel['num_workers'].iloc[0]
        efficiency = (speedup / workers) * 100
        logger.info(f"Eficiência paralela: {efficiency:.2f}%")
    
    return df_combined

def generate_visualizations(df_combined):
    """Gera visualizações comparativas"""
    
    # Cria diretório para gráficos
    plots_dir = Path("results/comparison_plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Comparação de tempo de execução
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_combined, x='algorithm', y='execution_time')
    plt.title('Tempo de Execução por Algoritmo')
    plt.ylabel('Tempo (segundos)')
    
    # 2. Comparação de taxa de redução
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_combined, x='algorithm', y='reduction_rate')
    plt.title('Taxa de Redução por Algoritmo')
    plt.ylabel('Taxa de Redução (%)')
    
    # 3. Relação instâncias originais vs tempo
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df_combined, x='original_instances', y='execution_time', 
                   hue='algorithm', alpha=0.7)
    plt.title('Instâncias Originais vs Tempo de Execução')
    plt.xlabel('Número de Instâncias Originais')
    plt.ylabel('Tempo (segundos)')
    
    # 4. Eficiência: instâncias processadas por segundo
    plt.subplot(2, 2, 4)
    if 'instances_per_second' in df_combined.columns:
        sns.barplot(data=df_combined, x='algorithm', y='instances_per_second')
        plt.title('Instâncias Processadas por Segundo')
        plt.ylabel('Instâncias/segundo')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico de barras detalhado
    plt.figure(figsize=(15, 10))
    
    # Métricas por dataset
    datasets = df_combined['dataset_name'].unique()
    if len(datasets) > 1:
        plt.subplot(2, 1, 1)
        pivot_time = df_combined.pivot(index='dataset_name', columns='algorithm', values='execution_time')
        pivot_time.plot(kind='bar', ax=plt.gca())
        plt.title('Tempo de Execução por Dataset')
        plt.ylabel('Tempo (segundos)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        pivot_reduction = df_combined.pivot(index='dataset_name', columns='algorithm', values='reduction_rate')
        pivot_reduction.plot(kind='bar', ax=plt.gca())
        plt.title('Taxa de Redução por Dataset')
        plt.ylabel('Taxa de Redução (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    logger.success(f"Gráficos salvos em {plots_dir}")

def generate_report(df_combined):
    """Gera relatório detalhado em CSV"""
    
    # Relatório de comparação
    comparison_stats = df_combined.groupby('algorithm').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'reduction_rate': ['mean', 'std', 'min', 'max'],
        'original_instances': ['mean', 'std'],
        'reduced_instances': ['mean', 'std']
    }).round(4)
    
    # Achata as colunas multi-nível
    comparison_stats.columns = ['_'.join(col).strip() for col in comparison_stats.columns]
    
    # Salva relatório
    reports_dir = Path("results/comparison_reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    comparison_stats.to_csv(reports_dir / f"algorithm_comparison_{timestamp}.csv")
    
    # Relatório detalhado
    df_combined.to_csv(reports_dir / f"detailed_results_{timestamp}.csv", index=False)
    
    logger.success(f"Relatórios salvos em {reports_dir}")
    
    return comparison_stats

def main():
    """Função principal"""
    logger.info("Iniciando comparação de algoritmos")
    
    # Carrega métricas
    df_sequential, df_parallel = load_latest_metrics()
    
    if df_sequential is None or df_parallel is None:
        return
    
    # Compara algoritmos
    df_combined = compare_algorithms(df_sequential, df_parallel)
    
    # Gera visualizações
    generate_visualizations(df_combined)
    
    # Gera relatório
    comparison_stats = generate_report(df_combined)
    
    # Mostra estatísticas finais
    logger.info("=== ESTATÍSTICAS FINAIS ===")
    print(comparison_stats)
    
    logger.success("Comparação completa!")

if __name__ == "__main__":
    main() 