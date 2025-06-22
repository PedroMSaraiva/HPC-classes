#!/usr/bin/env python3
"""
Script principal para executar análise completa de seleção de instâncias.
Executa algoritmos sequencial e paralelo, compara resultados e valida qualidade.
"""
import subprocess
import sys
import time
import os
from pathlib import Path
from rf_modules.logger_setup import setup_logger

# Configuração
logger = setup_logger()

def run_script(script_path, description):
    """Executa um script Python e monitora o resultado"""
    logger.info(f"Iniciando: {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hora de timeout
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.success(f"✓ {description} concluído em {execution_time:.2f}s")
            if result.stdout:
                logger.info("Saída:")
                print(result.stdout)
            return True
        else:
            logger.error(f"✗ {description} falhou após {execution_time:.2f}s")
            if result.stderr:
                logger.error("Erro:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} excedeu tempo limite (1 hora)")
        return False
    except Exception as e:
        logger.error(f"✗ Erro ao executar {description}: {e}")
        return False

def check_prerequisites():
    """Verifica se os pré-requisitos estão atendidos"""
    logger.info("Verificando pré-requisitos...")
    
    # Verifica se o diretório de datasets existe
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        logger.error(f"Diretório {dataset_dir.absolute()} não encontrado")
        return False
    
    # Busca arquivos CSV recursivamente nos subdiretórios (mesma lógica do DataLoader)
    csv_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.data') or file.endswith('.txt'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        logger.error(f"Nenhum arquivo CSV encontrado no diretório {dataset_dir.absolute()} e seus subdiretórios")
        return False
    
    logger.info(f"Encontrados {len(csv_files)} datasets para análise")
    
    # Verifica se os módulos necessários existem
    required_modules = [
        "rf_modules/logger_setup.py",
        "rf_modules/instance_selector.py"
    ]
    
    for module in required_modules:
        if not Path(module).exists():
            logger.error(f"Módulo necessário não encontrado: {module}")
            return False
    
    logger.success("Pré-requisitos atendidos")
    return True

def create_output_directories():
    """Cria diretórios de saída necessários"""
    directories = [
        "results",
        "results/instance_selection",
        "results/parallel_instance_selection",
        "results/reduced_datasets",
        "results/comparison_plots",
        "results/comparison_reports",
        "results/validation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
    
    logger.info("Diretórios de saída criados")

def main():
    """Função principal do pipeline completo"""
    logger.info("=== INICIANDO ANÁLISE COMPLETA DE SELEÇÃO DE INSTÂNCIAS ===")
    
    # Verifica pré-requisitos
    if not check_prerequisites():
        logger.error("Pré-requisitos não atendidos. Encerrando.")
        return False
    
    # Cria diretórios de saída
    create_output_directories()
    
    # Pipeline de execução
    pipeline = [
        ("run_instance_selection.py", "Seleção de Instâncias Sequencial"),
        ("run_parallel_instance_selection.py", "Seleção de Instâncias Paralela"),
        ("compare_algorithms.py", "Comparação de Algoritmos"),
        ("validate_reduced_datasets.py", "Validação dos Datasets Reduzidos")
    ]
    
    total_start_time = time.time()
    successful_steps = 0
    
    for script, description in pipeline:
        script_path = Path(script)
        
        if not script_path.exists():
            logger.error(f"Script não encontrado: {script}")
            continue
        
        success = run_script(script_path, description)
        if success:
            successful_steps += 1
        else:
            logger.warning(f"Falha em: {description}")
            # Continua mesmo com falhas para executar o que for possível
    
    total_time = time.time() - total_start_time
    
    # Relatório final
    logger.info("=== RELATÓRIO FINAL ===")
    logger.info(f"Etapas executadas com sucesso: {successful_steps}/{len(pipeline)}")
    logger.info(f"Tempo total de execução: {total_time:.2f}s")
    
    if successful_steps == len(pipeline):
        logger.success("✓ Pipeline completo executado com sucesso!")
        logger.info("\nResultados disponíveis em:")
        logger.info("- results/instance_selection/ - Resultados sequenciais")
        logger.info("- results/parallel_instance_selection/ - Resultados paralelos")
        logger.info("- results/reduced_datasets/ - Datasets reduzidos")
        logger.info("- results/comparison_plots/ - Gráficos comparativos")
        logger.info("- results/comparison_reports/ - Relatórios de comparação")
        logger.info("- results/validation/ - Validação de qualidade")
        return True
    else:
        logger.warning("⚠ Pipeline executado com algumas falhas")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 