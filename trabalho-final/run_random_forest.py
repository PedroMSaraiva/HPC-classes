#!/usr/bin/env python3
"""
Script para aplicação de Random Forest em múltiplos datasets.

Este script analisa automaticamente todos os datasets no diretório 'dataset',
aplica pré-processamento adequado, treina modelos Random Forest,
e gera relatórios e visualizações dos resultados.

Versão modular e otimizada com profiling de tempo.

Autor: Pedro Saraiva
Data: 2023
"""
import os
import warnings
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# Importa módulos personalizados
from rf_modules.logger_setup import setup_logger
from rf_modules.data_loader import DataLoader
from rf_modules.model import RandomForestModel
from rf_modules.visualization import Visualizer
from rf_modules.profiler import Profiler

# Configuração do logger
logger = setup_logger()

# Suprime avisos
warnings.filterwarnings('ignore')

# Constantes
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")
PROFILES_DIR = Path("results/profiling")

# Cria diretórios necessários
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
PROFILES_DIR.mkdir(exist_ok=True, parents=True)

# Número de threads para paralelização
NUM_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

# Inicializa objetos globais
data_loader = DataLoader(dataset_dir=DATASET_DIR)
visualizer = Visualizer(plots_dir=PLOTS_DIR)
profiler = Profiler(results_dir=PROFILES_DIR)


def process_dataset(file_path):
    """
    Processa um único dataset - função para uso em paralelização.
    
    Args:
        file_path (str): Caminho para o arquivo do dataset
        
    Returns:
        tuple: (dataset_info, metrics, feature_importance) ou None se falhar
    """
    try:
        dataset_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        full_name = f"{dataset_name}/{file_name}"
        
        logger.info(f"Processando {full_name}")
        
        # Mede tempo de carregamento do dataset
        start_time = time.time()
        df = data_loader.load_dataset(file_path)
        load_time = time.time() - start_time
        
        if df is None:
            logger.warning(f"Falha ao carregar {full_name}")
            return None
        
        logger.info(f"Dataset carregado com forma: {df.shape}")
        
        # Mede tempo de pré-processamento
        start_time = time.time()
        X, y, target_col, task_type = data_loader.preprocess_dataset(df)
        preprocess_time = time.time() - start_time
        
        if X is None:
            logger.warning(f"Falha ao pré-processar {full_name}")
            return None
        
        logger.info(f"Tipo de tarefa: {task_type}, Alvo: {target_col}")
        
        # Executa Random Forest
        dataset_id = f"{dataset_name}_{file_name.split('.')[0]}"
        
        # Divide os dados
        X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
        
        # Cria modelo
        rf_model = RandomForestModel()
        rf_model.create_model(task_type)
        
        # Treina o modelo
        rf_model.train(X_train, y_train)
        
        # Avalia o modelo
        metrics, y_pred = rf_model.evaluate(X_test, y_test, X, y)
        
        # Adiciona informações de tempo para reporting
        metrics['load_time'] = load_time
        metrics['preprocess_time'] = preprocess_time
        
        # Calcula importância das features
        feature_importance = RandomForestModel.calculate_feature_importance(rf_model.model, X)
        
        # Cria visualizações
        if task_type == 'classificação':
            n_classes = len(set(y))
            class_names = [f"Classe {i}" for i in range(n_classes)]
            visualizer.plot_confusion_matrix(y_test, y_pred, class_names, dataset_id, metrics['timestamp'])
        else:
            visualizer.plot_regression_results(y_test, y_pred, dataset_id, metrics['timestamp'])
        
        # Plota importância das features
        visualizer.plot_feature_importance(feature_importance, dataset_id, top_n=15, timestamp=metrics['timestamp'])
        
        dataset_info = {
            'full_name': full_name,
            'task_type': task_type,
            'target_column': target_col,
            'dataset_shape': df.shape
        }
        
        # Registra resultados
        if task_type == 'classificação':
            logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
            logger.info(f"Pontuação de validação cruzada: {metrics['cv_scores_mean']:.4f} ± {metrics['cv_scores_std']:.4f}")
        else:
            logger.info(f"MSE: {metrics['mse']:.4f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"Pontuação de validação cruzada: {-metrics['cv_scores_mean']:.4f} ± {metrics['cv_scores_std']:.4f}")
        
        logger.info(f"Top 5 features importantes: {feature_importance.head(5)['feature'].tolist()}")
        logger.info("-" * 80)
        
        return dataset_info, metrics, feature_importance
    except Exception as e:
        logger.exception(f"Erro ao processar {file_path}: {str(e)}")
        return None


@profiler.profile_function()
def main():
    """
    Função principal para processar todos os datasets.
    """
    start_time = time.time()
    logger.info("Iniciando análise de Random Forest em todos os datasets")
    
    # Encontra todos os arquivos CSV
    csv_files = data_loader.find_csv_files()
    logger.info(f"Encontrados {len(csv_files)} arquivos CSV")
    
    # Resultados
    results = {}
    
    # Processa cada dataset em paralelo
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submete todos os trabalhos
        future_to_file = {executor.submit(process_dataset, file_path): file_path for file_path in csv_files}
        
        # Processa os resultados à medida que são concluídos
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    dataset_info, metrics, feature_importance = result
                    full_name = dataset_info['full_name']
                    
                    results[full_name] = {
                        'metrics': metrics,
                        'feature_importance': feature_importance.to_dict() if feature_importance is not None else None,
                        'task_type': dataset_info['task_type'],
                        'target_column': dataset_info['target_column'],
                        'dataset_shape': dataset_info['dataset_shape']
                    }
            except Exception as e:
                logger.exception(f"Erro ao processar resultados para {file_path}: {str(e)}")
    
    logger.info(f"Processamento concluído para {len(results)} datasets")
    
    # Cria gráficos comparativos
    visualizer.create_comparative_plots(results)
    
    # Salva todos os resultados em um arquivo
    with open(RESULTS_DIR / "random_forest_summary.txt", 'w') as f:
        f.write("Resumo da Análise de Random Forest\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total de datasets processados: {len(results)}\n")
        f.write(f"Tempo total de execução: {time.time() - start_time:.2f} segundos\n\n")
        f.write("-" * 80 + "\n\n")
        
        for dataset, result in results.items():
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Forma: {result['dataset_shape']}\n")
            f.write(f"Tarefa: {result['task_type']}\n")
            f.write(f"Alvo: {result['target_column']}\n")
            
            # Informações de timestamp e tempo
            timestamp_str = result['metrics']['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if 'timestamp' in result['metrics'] else "N/A"
            f.write(f"Timestamp: {timestamp_str}\n")
            f.write(f"Tempo de carregamento: {result['metrics'].get('load_time', 0):.4f} segundos\n")
            f.write(f"Tempo de pré-processamento: {result['metrics'].get('preprocess_time', 0):.4f} segundos\n")
            f.write(f"Tempo de treinamento: {result['metrics'].get('training_time', 0):.4f} segundos\n")
            
            # Métricas de desempenho específicas por tipo de tarefa
            if result['task_type'] == 'classificação':
                f.write(f"Acurácia: {result['metrics']['accuracy']:.4f}\n")
                f.write(f"Pontuação de validação cruzada: {result['metrics']['cv_scores_mean']:.4f} ± {result['metrics']['cv_scores_std']:.4f}\n")
                f.write("\nRelatório de Classificação:\n")
                f.write(f"{result['metrics']['classification_report']}\n")
            else:
                f.write(f"MSE: {result['metrics']['mse']:.4f}\n")
                f.write(f"R²: {result['metrics']['r2']:.4f}\n")
                f.write(f"Pontuação de validação cruzada: {-result['metrics']['cv_scores_mean']:.4f} ± {result['metrics']['cv_scores_std']:.4f}\n")
            
            # Tratamento para lidar corretamente com o dicionário de feature_importance
            f.write("\nTop 5 Features:\n")
            if result['feature_importance'] is not None:
                # Obter as features e importâncias a partir da estrutura do dicionário
                feature_dict = result['feature_importance']
                # Estrutura do to_dict() tem 'feature' e 'importance' como chaves principais
                features = list(feature_dict['feature'].values())
                importances = list(feature_dict['importance'].values())
                
                # Pegar apenas os 5 primeiros itens (ou menos se não houver 5)
                top_count = min(5, len(features))
                for i in range(top_count):
                    f.write(f"{i+1}. {features[i]}: {importances[i]:.4f}\n")
            else:
                f.write("Não disponível\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
    
    # Salva dados de profiling
    profiler.save_profile_results("rf_profiling")
    
    # Finaliza a organização das visualizações
    visualizer.finalize()
    
    # Calcula tempo total de execução
    total_time = time.time() - start_time
    
    logger.success(f"Análise completa em {total_time:.2f} segundos")
    logger.success(f"Resultados salvos em {RESULTS_DIR}/random_forest_summary.txt")
    logger.success(f"Gráficos organizados em {visualizer.run_dir}")
    logger.success(f"Dados de profiling salvos em {PROFILES_DIR}/")
    
    # Cria link para os resultados HTML para fácil acesso
    html_results_link = RESULTS_DIR / "visualizacoes.html"
    with open(html_results_link, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0;url=../plots/index.html">
    <title>Redirecionando...</title>
</head>
<body>
    <p>Redirecionando para visualizações... 
       <a href="../plots/index.html">Clique aqui se não for redirecionado automaticamente</a>
    </p>
</body>
</html>""")
    
    logger.success(f"Acesse as visualizações em HTML: {html_results_link}")


if __name__ == "__main__":
    main()