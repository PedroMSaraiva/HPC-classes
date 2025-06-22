#!/usr/bin/env python3
"""
Script para validar a qualidade dos datasets reduzidos comparando performance de classificação.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from rf_modules.logger_setup import setup_logger

# Configuração
logger = setup_logger()

def load_datasets():
    """Carrega datasets originais e reduzidos"""
    original_dir = Path("dataset")
    reduced_dir = Path("results/reduced_datasets")
    
    datasets = {}
    
    # Carrega datasets originais
    for csv_file in original_dir.glob("*.csv"):
        dataset_name = csv_file.stem
        df = pd.read_csv(csv_file)
        datasets[dataset_name] = {'original': df}
        logger.info(f"Dataset original carregado: {dataset_name} ({len(df)} instâncias)")
    
    # Carrega datasets reduzidos
    for csv_file in reduced_dir.glob("*.csv"):
        # Extrai nome do dataset original do nome do arquivo
        filename = csv_file.stem
        if filename.startswith("reduced_"):
            dataset_name = filename.replace("reduced_", "")
            if dataset_name in datasets:
                df_reduced = pd.read_csv(csv_file)
                datasets[dataset_name]['reduced'] = df_reduced
                logger.info(f"Dataset reduzido carregado: {dataset_name} ({len(df_reduced)} instâncias)")
    
    # Remove datasets que não têm versão reduzida
    complete_datasets = {k: v for k, v in datasets.items() if 'reduced' in v}
    
    logger.info(f"Datasets completos para validação: {len(complete_datasets)}")
    return complete_datasets

def prepare_data(df):
    """Prepara dados para classificação"""
    # Assume que a primeira coluna é o target
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Normaliza features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def evaluate_dataset(X, y, dataset_name, dataset_type):
    """Avalia performance de um dataset"""
    
    # Configuração do classificador
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Mede tempo de treinamento
    start_time = time.time()
    
    # Scores de cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    training_time = time.time() - start_time
    
    # Treina modelo final para métricas detalhadas
    rf.fit(X, y)
    y_pred = rf.predict(X)
    
    metrics = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'num_instances': len(X),
        'num_features': X.shape[1],
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'training_time': training_time,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
    }
    
    logger.info(f"{dataset_type} {dataset_name}: "
               f"Acc={metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}, "
               f"Time={training_time:.2f}s, Instances={len(X)}")
    
    return metrics

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

def main():
    """Função principal"""
    logger.info("Iniciando validação dos datasets reduzidos")
    
    # Carrega datasets
    datasets = load_datasets()
    
    if not datasets:
        logger.error("Nenhum dataset encontrado para validação")
        return
    
    all_metrics = []
    comparisons = []
    
    # Avalia cada dataset
    for dataset_name, data in datasets.items():
        logger.info(f"Validando dataset: {dataset_name}")
        
        # Prepara dados originais
        X_orig, y_orig = prepare_data(data['original'])
        orig_metrics = evaluate_dataset(X_orig, y_orig, dataset_name, 'original')
        all_metrics.append(orig_metrics)
        
        # Prepara dados reduzidos
        X_red, y_red = prepare_data(data['reduced'])
        red_metrics = evaluate_dataset(X_red, y_red, dataset_name, 'reduced')
        all_metrics.append(red_metrics)
        
        # Compara resultados
        comparison = compare_datasets(orig_metrics, red_metrics)
        comparisons.append(comparison)
        
        logger.info(f"Dataset {dataset_name}: Redução {comparison['reduction_rate']:.1f}%, "
                   f"Mudança na acurácia: {comparison['accuracy_change']:+.4f}")
    
    # Gera relatório
    summary, df_comparisons = generate_validation_report(all_metrics, comparisons)
    
    # Imprime resumo
    print_validation_summary(summary, df_comparisons)
    
    logger.success("Validação completa!")

if __name__ == "__main__":
    main() 