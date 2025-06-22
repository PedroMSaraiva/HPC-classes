#!/usr/bin/env python3
"""
Script para validar a qualidade dos datasets reduzidos comparando performance de classificação.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from rf_modules.logger_setup import setup_logger
import warnings
from datetime import datetime

# Configuração
logger = setup_logger()

def load_datasets():
    """Carrega datasets originais e reduzidos"""
    original_dir = Path("dataset")
    # Procura datasets reduzidos nos diretórios corretos
    reduced_dirs = [
        Path("results/instance_selection/reduced_datasets"),
        Path("results/parallel_instance_selection/reduced_datasets"),
        Path("results/reduced_datasets")  # Mantém o diretório original como fallback
    ]
    
    datasets = {}
    
    # Busca arquivos CSV recursivamente nos subdiretórios (mesma lógica do DataLoader)
    original_files = []
    for root, dirs, files in os.walk(original_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.data') or file.endswith('.txt'):
                original_files.append(os.path.join(root, file))
    
    # Carrega datasets originais
    for csv_file_path in original_files:
        csv_file = Path(csv_file_path)
        dataset_name = csv_file.stem
        df = pd.read_csv(csv_file_path)
        datasets[dataset_name] = {'original': df}
        logger.info(f"Dataset original carregado: {dataset_name} ({len(df)} instâncias)")
    
    # Carrega datasets reduzidos de todos os diretórios possíveis
    for reduced_dir in reduced_dirs:
        if reduced_dir.exists():
            for csv_file in reduced_dir.glob("*.csv"):
                # Extrai nome do dataset original do nome do arquivo
                filename = csv_file.stem
                
                # Remove prefixos e sufixos para encontrar o nome do dataset original
                dataset_name = filename
                if filename.startswith("reduced_"):
                    dataset_name = filename.replace("reduced_", "")
                
                # Remove sufixos como _reduced_sequential, _reduced_parallel, etc.
                for suffix in ["_reduced_sequential", "_reduced_parallel", "_reduced"]:
                    if dataset_name.endswith(suffix):
                        dataset_name = dataset_name.replace(suffix, "")
                        break
                
                # Verifica se o dataset original existe
                if dataset_name in datasets:
                    df_reduced = pd.read_csv(csv_file)
                    
                    # Se já existe uma versão reduzida, usa a mais recente ou a com mais instâncias
                    if 'reduced' not in datasets[dataset_name]:
                        datasets[dataset_name]['reduced'] = df_reduced
                        logger.info(f"Dataset reduzido carregado: {dataset_name} ({len(df_reduced)} instâncias) - {csv_file.name}")
                    else:
                        # Compara e mantém o dataset com mais instâncias (mais conservador)
                        if len(df_reduced) > len(datasets[dataset_name]['reduced']):
                            datasets[dataset_name]['reduced'] = df_reduced
                            logger.info(f"Dataset reduzido atualizado: {dataset_name} ({len(df_reduced)} instâncias) - {csv_file.name}")
    
    # Remove datasets que não têm versão reduzida
    complete_datasets = {k: v for k, v in datasets.items() if 'reduced' in v}
    
    logger.info(f"Datasets completos para validação: {len(complete_datasets)}")
    return complete_datasets

def prepare_data(df):
    """Prepara dados para treinamento"""
    # Separar features e target
    X = df.iloc[:, 1:]  # Todas as colunas exceto a primeira (target)
    y = df.iloc[:, 0]   # Primeira coluna é o target
    
    # Converter variáveis categóricas para numéricas
    label_encoders = {}
    
    # Processar target se for categórico
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    # Processar features categóricas
    for col in X.columns:
        if X[col].dtype == 'object' or isinstance(X[col].iloc[0], str):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Converter para valores numéricos
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Remover linhas com valores NaN
    mask = ~(X.isna().any(axis=1) | pd.isna(y))
    X = X[mask]
    y = y[mask]
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def evaluate_dataset(X, y, dataset_name, dataset_type):
    """Avalia performance de um dataset"""
    
    # Verificar se há dados suficientes
    if len(X) < 10:
        logger.warning(f"Dataset {dataset_name} ({dataset_type}) muito pequeno: {len(X)} instâncias")
        return None
    
    # Configuração do classificador
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Determina número de folds baseado no tamanho do dataset e classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = min(class_counts)
    
    # Calcula número de folds adequado
    max_folds = min(5, len(X) // 20)  # Pelo menos 20 amostras por fold
    n_splits = min(max_folds, min_class_count)
    n_splits = max(2, n_splits)  # Mínimo de 2 folds
    
    # Se ainda não é possível fazer CV, usa holdout simples
    use_cv = n_splits >= 2 and min_class_count >= n_splits
    
    # Mede tempo de treinamento
    start_time = time.time()
    
    try:
        if use_cv:
            # Cross-validation estratificada
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            cv_accuracy_mean = cv_scores.mean()
            cv_accuracy_std = cv_scores.std()
        else:
            # Fallback: treina e testa no mesmo conjunto
            rf.fit(X, y)
            y_pred = rf.predict(X)
            cv_accuracy_mean = accuracy_score(y, y_pred)
            cv_accuracy_std = 0.0
            n_splits = 1
        
        training_time = time.time() - start_time
        
        # Treina modelo final para métricas detalhadas
        rf.fit(X, y)
        y_pred = rf.predict(X)
        
        metrics = {
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'num_instances': len(X),
            'num_features': X.shape[1],
            'num_classes': len(unique_classes),
            'min_class_count': min_class_count,
            'cv_folds': n_splits,
            'cv_accuracy_mean': cv_accuracy_mean,
            'cv_accuracy_std': cv_accuracy_std,
            'training_time': training_time,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"{dataset_type} {dataset_name}: "
                   f"Acc={cv_accuracy_mean:.4f}±{cv_accuracy_std:.4f}, "
                   f"Time={training_time:.2f}s, Instances={len(X)}, "
                   f"Classes={len(unique_classes)}, MinClass={min_class_count}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erro ao avaliar dataset {dataset_name} ({dataset_type}): {e}")
        
        try:
            # Fallback simples
            rf.fit(X, y)
            y_pred = rf.predict(X)
            
            metrics = {
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'num_instances': len(X),
                'num_features': X.shape[1],
                'num_classes': len(unique_classes),
                'min_class_count': min_class_count,
                'cv_folds': 0,
                'cv_accuracy_mean': accuracy_score(y, y_pred),
                'cv_accuracy_std': 0.0,
                'training_time': 0.0,
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            return metrics
            
        except Exception as e2:
            logger.error(f"Erro crítico ao avaliar {dataset_name} ({dataset_type}): {e2}")
            return None

def compare_datasets(original_metrics, reduced_metrics):
    """Compara métricas entre datasets originais e reduzidos"""
    
    comparison = {
        'dataset_name': original_metrics['dataset_name'],
        'original_instances': original_metrics['num_instances'],
        'reduced_instances': reduced_metrics['num_instances'],
        'reduction_rate': ((original_metrics['num_instances'] - reduced_metrics['num_instances']) 
                          / original_metrics['num_instances']) * 100,
        'original_accuracy': original_metrics['cv_accuracy_mean'],
        'reduced_accuracy': reduced_metrics['cv_accuracy_mean'],
        'accuracy_change': reduced_metrics['cv_accuracy_mean'] - original_metrics['cv_accuracy_mean'],
        'accuracy_change_pct': ((reduced_metrics['cv_accuracy_mean'] - original_metrics['cv_accuracy_mean']) 
                               / original_metrics['cv_accuracy_mean']) * 100,
        'original_training_time': original_metrics['training_time'],
        'reduced_training_time': reduced_metrics['training_time'],
        'speedup': original_metrics['training_time'] / reduced_metrics['training_time'] if reduced_metrics['training_time'] > 0 else 0,
        'original_f1': original_metrics['f1_score'],
        'reduced_f1': reduced_metrics['f1_score'],
        'f1_change': reduced_metrics['f1_score'] - original_metrics['f1_score']
    }
    
    return comparison

def generate_validation_report(all_metrics, comparisons):
    """Gera relatório de validação"""
    
    # Cria diretório para resultados
    validation_dir = Path("results/validation")
    validation_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Salva métricas detalhadas
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(validation_dir / f"detailed_metrics_{timestamp}.csv", index=False)
    
    # Salva comparações
    df_comparisons = pd.DataFrame(comparisons)
    df_comparisons.to_csv(validation_dir / f"dataset_comparisons_{timestamp}.csv", index=False)
    
    # Relatório resumido
    summary = {
        'total_datasets': len(comparisons),
        'avg_reduction_rate': df_comparisons['reduction_rate'].mean(),
        'avg_accuracy_change': df_comparisons['accuracy_change'].mean(),
        'avg_speedup': df_comparisons['speedup'].mean(),
        'datasets_with_improved_accuracy': len(df_comparisons[df_comparisons['accuracy_change'] > 0]),
        'datasets_with_maintained_accuracy': len(df_comparisons[abs(df_comparisons['accuracy_change']) <= 0.01]),
        'datasets_with_significant_speedup': len(df_comparisons[df_comparisons['speedup'] > 1.5])
    }
    
    # Salva resumo
    pd.DataFrame([summary]).to_csv(validation_dir / f"validation_summary_{timestamp}.csv", index=False)
    
    logger.success(f"Relatórios de validação salvos em {validation_dir}")
    
    return summary, df_comparisons

def print_validation_summary(summary, df_comparisons):
    """Imprime resumo da validação"""
    
    logger.info("=== RESUMO DA VALIDAÇÃO ===")
    logger.info(f"Total de datasets validados: {summary['total_datasets']}")
    logger.info(f"Taxa de redução média: {summary['avg_reduction_rate']:.2f}%")
    logger.info(f"Mudança média na acurácia: {summary['avg_accuracy_change']:.4f}")
    logger.info(f"Speedup médio: {summary['avg_speedup']:.2f}x")
    logger.info(f"Datasets com acurácia melhorada: {summary['datasets_with_improved_accuracy']}")
    logger.info(f"Datasets com acurácia mantida (±1%): {summary['datasets_with_maintained_accuracy']}")
    logger.info(f"Datasets com speedup significativo (>1.5x): {summary['datasets_with_significant_speedup']}")
    
    logger.info("\n=== RESULTADOS POR DATASET ===")
    for _, row in df_comparisons.iterrows():
        status = "✓" if row['accuracy_change'] >= -0.01 else "⚠"
        logger.info(f"{status} {row['dataset_name']}: "
                   f"Redução {row['reduction_rate']:.1f}%, "
                   f"Acc {row['accuracy_change']:+.4f}, "
                   f"Speedup {row['speedup']:.2f}x")

def save_results(results):
    """Salva resultados da validação"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cria DataFrame
    df_results = pd.DataFrame(results)
    
    # Define diretório de resultados
    results_dir = "results/validation"
    os.makedirs(results_dir, exist_ok=True)
    
    # Salva CSV
    results_file = os.path.join(results_dir, f"validation_results_{timestamp}.csv")
    df_results.to_csv(results_file, index=False)
    
    logger.info(f"Resultados salvos em: {results_file}")
    
    return results_file

def main():
    """Função principal"""
    logger.info("Iniciando validação dos datasets reduzidos")
    
    # Carrega datasets
    datasets = load_datasets()
    
    if not datasets:
        logger.error("Nenhum dataset encontrado para validação")
        return
    
    logger.info(f"Datasets completos para validação: {len(datasets)}")
    
    all_results = []
    
    for dataset_name, data in datasets.items():
        logger.info(f"Validando dataset: {dataset_name}")
        
        try:
            # Prepara dados originais
            X_orig, y_orig = prepare_data(data['original'])
            
            # Prepara dados reduzidos
            X_red, y_red = prepare_data(data['reduced'])
            
            # Avalia datasets
            orig_metrics = evaluate_dataset(X_orig, y_orig, dataset_name, "original")
            red_metrics = evaluate_dataset(X_red, y_red, dataset_name, "reduced")
            
            # Verifica se ambas avaliações foram bem-sucedidas
            if orig_metrics is None or red_metrics is None:
                logger.warning(f"Pulando dataset {dataset_name} devido a problemas na avaliação")
                continue
            
            # Compara resultados
            comparison = compare_datasets(orig_metrics, red_metrics)
            
            # Armazena resultados
            result = {
                'dataset_name': dataset_name,
                'original_instances': orig_metrics['num_instances'],
                'reduced_instances': red_metrics['num_instances'],
                'reduction_rate': comparison['reduction_rate'],
                'original_accuracy': orig_metrics['cv_accuracy_mean'],
                'reduced_accuracy': red_metrics['cv_accuracy_mean'],
                'accuracy_change': comparison['accuracy_change'],
                'original_training_time': orig_metrics['training_time'],
                'reduced_training_time': red_metrics['training_time'],
                'speedup': comparison['speedup'],
                'original_classes': orig_metrics['num_classes'],
                'reduced_classes': red_metrics['num_classes'],
                'original_min_class': orig_metrics['min_class_count'],
                'reduced_min_class': red_metrics['min_class_count']
            }
            
            all_results.append(result)
            
            logger.info(f"Dataset {dataset_name}: "
                       f"Redução {comparison['reduction_rate']:.1f}%, "
                       f"Mudança na acurácia: {comparison['accuracy_change']:+.4f}")
            
        except Exception as e:
            logger.error(f"Erro ao processar dataset {dataset_name}: {e}")
            continue
    
    # Salva resultados
    if all_results:
        save_results(all_results)
        
        # Estatísticas gerais
        logger.info(f"\n=== RESUMO GERAL ===")
        logger.info(f"Datasets validados: {len(all_results)}")
        
        avg_reduction = np.mean([r['reduction_rate'] for r in all_results])
        avg_accuracy_change = np.mean([r['accuracy_change'] for r in all_results])
        avg_speedup = np.mean([r['speedup'] for r in all_results if r['speedup'] > 0])
        
        logger.info(f"Redução média: {avg_reduction:.1f}%")
        logger.info(f"Mudança média na acurácia: {avg_accuracy_change:+.4f}")
        logger.info(f"Speedup médio: {avg_speedup:.2f}x")
        
        positive_changes = sum(1 for r in all_results if r['accuracy_change'] > 0)
        logger.info(f"Datasets com melhoria na acurácia: {positive_changes}/{len(all_results)}")
        
    else:
        logger.warning("Nenhum resultado válido foi gerado")
    
    logger.info("Validação concluída")

if __name__ == "__main__":
    main() 