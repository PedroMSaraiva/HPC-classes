#!/usr/bin/env python3
"""
Script para seleção de instâncias usando Algoritmo Genético e comparação com resultados originais.
Este script aplica seleção de instâncias em datasets de classificação usando AG,

"""
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import random
from datetime import datetime
import time

# Importa módulos personalizados
from rf_modules.logger_setup import setup_logger
from rf_modules.data_loader import DataLoader
from rf_modules.instance_selector import InstanceSelector

# Configuração do logger
logger = setup_logger()

# Suprime avisos
warnings.filterwarnings('ignore')

# Constantes
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
INSTANCE_SEL_DIR = Path("results/instance_selection")
REDUCED_DATASETS_DIR = Path("results/instance_selection/reduced_datasets")

# Cria diretórios necessários
for dir_path in [RESULTS_DIR, INSTANCE_SEL_DIR, REDUCED_DATASETS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

def process_dataset_with_instance_selection(file_path, data_loader):
    """
    Processa um dataset aplicando seleção de instâncias e Random Forest
    
    Args:
        file_path: Caminho do arquivo do dataset
        data_loader: Instância do DataLoader
        
    Returns:
        dict: Resultados da análise
    """
    try:
        dataset_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        full_name = f"{dataset_name}/{file_name}"
        
        logger.info(f"Processando {full_name} com seleção de instâncias")
        
        # Carrega e pré-processa o dataset
        df = data_loader.load_dataset(file_path)
        if df is None:
            return None
            
        X, y, target_col, task_type = data_loader.preprocess_dataset(df)
   
        if X is None or task_type != 'classificação':
            return None
            
        # Métricas do dataset original
        original_instances = len(X)
        original_features = X.shape[1]
        original_classes = len(np.unique(y))
        
        # Aplica seleção de instâncias no conjunto de treino
        start_time = time.time()
        instance_selector = InstanceSelector(
            num_generations=50,
            population_size=20,
            num_parents_mating=4
        )
        
        X_selected, y_selected = instance_selector.select_instances(X, y)
        execution_time = time.time() - start_time
            
        if X_selected is None:
            return None
            
        # Métricas do dataset reduzido
        reduced_instances = len(X_selected)
        reduction_rate = (1 - reduced_instances / original_instances) * 100
        best_fitness = instance_selector.best_fitness
        
        # Salva o dataset reduzido
        dataset_reduced = pd.concat([y_selected.reset_index(drop=True), 
                                   X_selected.reset_index(drop=True)], axis=1)
        
        # Nome do arquivo baseado no dataset original
        base_name = Path(file_path).stem
        reduced_file_name = f"{base_name}_reduced_sequential.csv"
        reduced_file_path = REDUCED_DATASETS_DIR / reduced_file_name
        dataset_reduced.to_csv(reduced_file_path, index=False)
        
        # Registra resultados com métricas detalhadas
        results = {
            'dataset_name': full_name,
            'algorithm': 'Sequential_GA',
            'original_instances': original_instances,
            'reduced_instances': reduced_instances,
            'reduction_rate': reduction_rate,
            'original_features': original_features,
            'original_classes': original_classes,
            'execution_time': execution_time,
            'best_fitness': best_fitness,
            'reduced_dataset_path': str(reduced_file_path),
            'X': X_selected,
            'Y': y_selected,
        }

        logger.info(f"Resultados para {full_name}:")
        logger.info(f"  Instâncias originais: {original_instances}")
        logger.info(f"  Instâncias reduzidas: {reduced_instances}")
        logger.info(f"  Taxa de redução: {reduction_rate:.2f}%")
        logger.info(f"  Melhor fitness: {best_fitness:.4f}")
        logger.info(f"  Tempo de execução: {execution_time:.2f}s")
        logger.info(f"  Dataset reduzido salvo em: {reduced_file_path}")
        
        return results
        
    except Exception as e:
        logger.exception(f"Erro ao processar {file_path}: {str(e)}")
        return None

def main():
    """Função principal"""
    start_time = datetime.now()
    logger.info("Iniciando análise com seleção de instâncias")
    
    # Inicializa componentes
    data_loader = DataLoader(dataset_dir=DATASET_DIR)
    
    # Encontra datasets de classificação
    csv_files = data_loader.find_csv_files()
    results = []
    
    # Processa cada dataset
    for file_path in csv_files:
        result = process_dataset_with_instance_selection(file_path, data_loader)
        if result is not None:
            results.append(result)
    
    # Salva métricas de desempenho em CSV
    if results:
        # Cria DataFrame com métricas de performance
        metrics_data = []
        for result in results:
            metrics_data.append({
                'dataset_name': result['dataset_name'],
                'algorithm': result['algorithm'],
                'original_instances': result['original_instances'],
                'reduced_instances': result['reduced_instances'],
                'reduction_rate': result['reduction_rate'],
                'original_features': result['original_features'],
                'original_classes': result['original_classes'],
                'execution_time': result['execution_time'],
                'best_fitness': result['best_fitness'],
                'reduced_dataset_path': result['reduced_dataset_path']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = INSTANCE_SEL_DIR / f"sequential_performance_metrics_{timestamp}.csv"
        df_metrics.to_csv(metrics_file, index=False)
        
        logger.success(f"Métricas de performance salvos em {metrics_file}")
        
        # Salva também o último dataset processado como exemplo
        if results:
            last_result = results[-1]
            df_results = pd.concat([
                last_result['Y'].reset_index(drop=True),
                last_result['X'].reset_index(drop=True)
                ], axis=1)
            numero = random.randint(1, 1000)
            results_file = INSTANCE_SEL_DIR / f"instance_selection_results_{numero}.csv"
            df_results.to_csv(results_file, index=False)
            logger.success(f"Último resultado salvo em {results_file}")
        
    # Finaliza
    elapsed_time = datetime.now() - start_time
    logger.success(f"Análise completa em {elapsed_time}")
    logger.success(f"Resultados disponíveis em {INSTANCE_SEL_DIR}")
    logger.success(f"Datasets reduzidos disponíveis em {REDUCED_DATASETS_DIR}")

if __name__ == "__main__":
    main() 