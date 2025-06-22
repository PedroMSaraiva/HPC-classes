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

# Cria diretórios necessários
for dir_path in [RESULTS_DIR, INSTANCE_SEL_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

def process_dataset_with_instance_selection(file_path, data_loader):
    """
    Processa um dataset aplicando seleção de instâncias e Random Forest
    
    Args:
        file_path: Caminho do arquivo do dataset
        data_loader: Instância do DataLoader
        visualizer: Instância do Visualizer
        
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
            

        # Aplica seleção de instâncias no conjunto de treino
        instance_selector = InstanceSelector(
            num_generations=50,
            population_size=20,
            num_parents_mating=4
        )
        
        X_selected, y_selected, = \
            instance_selector.select_instances(X, y)
            
        if X_selected is None:
            return None
            
        # Registra resultados
        results = {
            'X': X_selected,
            'Y': y_selected,
        }

        # Calcula redução de instâncias
        reduction_rate = (1 - len(X_selected) / len(X)) * 100

        logger.info(f"Resultados para {full_name}:")
        logger.info(f"Redução de instâncias: {reduction_rate:.2f}%")
    
        
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
        result = process_dataset_with_instance_selection(file_path, data_loader,)
        if result is not None:
            results.append(result)
    
    # Salva resultados em CSV
    if results:
        df_results = pd.concat([
            results[0]['Y'].reset_index(drop=True),
            results[0]['X'].reset_index(drop=True)
            ], axis=1)
        numero = random.randint(1, 1000)
        results_file = INSTANCE_SEL_DIR / f"parallel_instance_selection_results_{numero}.csv"
        df_results.to_csv(results_file, index=False)
        logger.success(f"Resultados salvos em {results_file}")
        
    # Finaliza
    elapsed_time = datetime.now() - start_time
    logger.success(f"Análise completa em {elapsed_time}")
    logger.success(f"Resultados disponíveis em {INSTANCE_SEL_DIR}")

if __name__ == "__main__":
    main() 