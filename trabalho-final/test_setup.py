#!/usr/bin/env python3
"""
Script de teste para verificar se o ambiente está configurado corretamente.
"""
import sys
import os
import importlib
from pathlib import Path

def test_python_version():
    """Testa se a versão do Python é adequada"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário")
        return False
    print("✅ Versão do Python adequada")
    return True

def test_dependencies():
    """Testa se as dependências estão instaladas"""
    required_packages = [
        'numpy',
        'pandas', 
        'sklearn',
        'matplotlib',
        'seaborn',
        'pygad',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - FALTANDO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nPacotes faltando: {', '.join(missing_packages)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    return True

def test_directory_structure():
    """Testa se a estrutura de diretórios está correta"""
    required_dirs = [
        'rf_modules',
        'dataset'
    ]
    
    required_files = [
        'rf_modules/logger_setup.py',
        'rf_modules/instance_selector.py',
        'requirements.txt'
    ]
    
    all_good = True
    
    # Testa diretórios
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Diretório {directory} - OK")
        else:
            print(f"❌ Diretório {directory} - FALTANDO")
            all_good = False
    
    # Testa arquivos
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ Arquivo {file_path} - OK")
        else:
            print(f"❌ Arquivo {file_path} - FALTANDO")
            all_good = False
    
    return all_good

def test_datasets():
    """Testa se há datasets disponíveis"""
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        print("❌ Diretório dataset não existe")
        return False
    
    # Busca arquivos CSV recursivamente nos subdiretórios (mesma lógica do DataLoader)
    csv_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.data') or file.endswith('.txt'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("⚠️  Nenhum arquivo CSV encontrado no diretório dataset e subdiretórios")
        print("   Coloque pelo menos um dataset CSV para testar")
        return False
    
    print(f"✅ Encontrados {len(csv_files)} datasets:")
    for csv_file in csv_files[:5]:  # Mostra apenas os primeiros 5
        print(f"   - {os.path.basename(csv_file)}")
    
    if len(csv_files) > 5:
        print(f"   ... e mais {len(csv_files) - 5} datasets")
    
    return True

def test_modules():
    """Testa se os módulos customizados podem ser importados"""
    try:
        from rf_modules.logger_setup import setup_logger
        print("✅ Módulo logger_setup - OK")
        
        from rf_modules.instance_selector import InstanceSelector
        print("✅ Módulo instance_selector - OK")
        
        # Testa se o logger funciona
        logger = setup_logger()
        logger.info("Teste do logger - OK")
        print("✅ Logger funcionando - OK")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao importar módulos: {e}")
        return False

def main():
    """Função principal de teste"""
    print("=== TESTE DE CONFIGURAÇÃO DO AMBIENTE ===\n")
    
    tests = [
        ("Versão do Python", test_python_version),
        ("Dependências", test_dependencies),
        ("Estrutura de diretórios", test_directory_structure),
        ("Datasets", test_datasets),
        ("Módulos customizados", test_modules)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Relatório final
    print("\n=== RELATÓRIO FINAL ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\nTestes: {passed}/{total} passaram")
    
    if passed == total:
        print("\n🎉 Ambiente configurado corretamente!")
        print("Você pode executar: python run_complete_analysis.py")
        return True
    else:
        print("\n⚠️  Alguns testes falharam. Verifique os erros acima.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 