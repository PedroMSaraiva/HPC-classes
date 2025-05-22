#!/usr/bin/env python3
"""
Script para seleção de instâncias usando Algoritmo Genético e comparação com resultados originais.

Este script aplica seleção de instâncias em datasets de classificação usando AG,
treina modelos Random Forest nos datasets reduzidos e compara com os resultados originais.

Autor: Pedro Saraiva
Data: 2024
"""
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Importa módulos personalizados
from rf_modules.logger_setup import setup_logger
from rf_modules.data_loader import DataLoader
from rf_modules.model import RandomForestModel
from rf_modules.visualization import Visualizer
from rf_modules.instance_selector import InstanceSelector

# Configuração do logger
logger = setup_logger()

# Suprime avisos
warnings.filterwarnings('ignore')

# Constantes
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")
INSTANCE_SEL_DIR = Path("results/instance_selection")

# Cria diretórios necessários
for dir_path in [RESULTS_DIR, PLOTS_DIR, INSTANCE_SEL_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

def process_dataset_with_instance_selection(file_path, data_loader, visualizer):
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
            
        # Divide dados em treino e teste para avaliação final
        X_train_full, X_test, y_train_full, y_test = data_loader.split_data(X, y)
        
        # Aplica seleção de instâncias no conjunto de treino
        instance_selector = InstanceSelector(
            num_generations=50,
            population_size=20,
            num_parents_mating=4
        )
        
        X_train_selected, y_train_selected, selected_indices, fitness_history = \
            instance_selector.select_instances(X_train_full, y_train_full)
            
        if X_train_selected is None:
            return None
            
        # Treina e avalia modelo com dados originais
        rf_original = RandomForestModel()
        rf_original.create_model(task_type)
        rf_original.train(X_train_full, y_train_full)
        metrics_original, _ = rf_original.evaluate(X_test, y_test, X, y)
        
        # Treina e avalia modelo com dados selecionados
        rf_selected = RandomForestModel()
        rf_selected.create_model(task_type)
        rf_selected.train(X_train_selected, y_train_selected)
        metrics_selected, _ = rf_selected.evaluate(X_test, y_test, X, y)
        
        # Calcula redução de instâncias
        reduction_rate = (1 - len(X_train_selected) / len(X_train_full)) * 100
        
        # Registra resultados
        results = {
            'dataset_name': full_name,
            'original_instances': len(X_train_full),
            'selected_instances': len(X_train_selected),
            'reduction_rate': reduction_rate,
            'original_accuracy': metrics_original['accuracy'],
            'selected_accuracy': metrics_selected['accuracy'],
            'accuracy_difference': metrics_selected['accuracy'] - metrics_original['accuracy'],
            'fitness_history': fitness_history
        }
        
        # Plota evolução do fitness
        visualizer.plot_fitness_evolution(
            fitness_history,
            dataset_name=full_name,
            save_dir=INSTANCE_SEL_DIR
        )
        
        # Plota comparação de acurácias
        visualizer.plot_accuracy_comparison(
            results,
            dataset_name=full_name,
            save_dir=INSTANCE_SEL_DIR
        )
        
        logger.info(f"Resultados para {full_name}:")
        logger.info(f"Redução de instâncias: {reduction_rate:.2f}%")
        logger.info(f"Acurácia original: {metrics_original['accuracy']:.4f}")
        logger.info(f"Acurácia com seleção: {metrics_selected['accuracy']:.4f}")
        
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
    visualizer = Visualizer(plots_dir=PLOTS_DIR)
    
    # Encontra datasets de classificação
    csv_files = data_loader.find_csv_files()
    results = []
    
    # Processa cada dataset
    for file_path in csv_files:
        result = process_dataset_with_instance_selection(file_path, data_loader, visualizer)
        if result is not None:
            results.append(result)
    
    # Salva resultados em CSV
    if results:
        df_results = pd.DataFrame(results)
        results_file = INSTANCE_SEL_DIR / "instance_selection_results.csv"
        df_results.to_csv(results_file, index=False)
        logger.success(f"Resultados salvos em {results_file}")
        
        # Cria visualizações comparativas
        visualizer.plot_overall_comparison(df_results, save_dir=INSTANCE_SEL_DIR)
    
    # Finaliza
    elapsed_time = datetime.now() - start_time
    logger.success(f"Análise completa em {elapsed_time}")
    logger.success(f"Resultados e visualizações disponíveis em {INSTANCE_SEL_DIR}")

if __name__ == "__main__":
    main() 