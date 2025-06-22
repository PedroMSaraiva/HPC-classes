"""
Módulo para carregamento e processamento de dados
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger

class DataLoader:
    """
    Classe para carregar e processar datasets
    """
    
    def __init__(self, dataset_dir="dataset"):
        """
        Inicializa o carregador de dados
        
        Args:
            dataset_dir (str): Diretório onde se encontram os datasets
        """
        self.dataset_dir = Path(dataset_dir)
        
    def find_csv_files(self):
        """
        Encontra todos os arquivos CSV no diretório de datasets e seus subdiretórios.
        
        Returns:
            list: Lista de caminhos para todos os arquivos CSV encontrados
        """
        logger.info(f"Buscando arquivos em {self.dataset_dir}")
        all_csv_files = []
        
        # Percorre o diretório de datasets
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.data') or file.endswith('.txt'):
                    all_csv_files.append(os.path.join(root, file))
        
        logger.info(f"Encontrados {len(all_csv_files)} arquivos para processamento")
        return all_csv_files
    
    def load_dataset(self, file_path):
        """
        Carrega um dataset a partir do caminho do arquivo, tentando diferentes delimitadores e codificações.
        
        Args:
            file_path (str): Caminho para o arquivo do dataset
            
        Returns:
            DataFrame: DataFrame pandas com os dados carregados ou None se falhar
        """
        logger.debug(f"Tentando carregar {file_path}")
        try:
            # Tenta delimitador vírgula primeiro
            df = pd.read_csv(file_path, index_col=0)
            logger.debug(f"Arquivo carregado com delimitador vírgula: {file_path}")
            return df
        except:
            try:
                # Tenta delimitador ponto-e-vírgula
                df = pd.read_csv(file_path, sep=';')
                logger.debug(f"Arquivo carregado com delimitador ponto-e-vírgula: {file_path}")
                return df
            except:
                try:
                    # Tenta delimitador tab
                    df = pd.read_csv(file_path, sep='\t')
                    logger.debug(f"Arquivo carregado com delimitador tab: {file_path}")
                    return df
                except:
                    try:
                        # Tenta delimitador espaço
                        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
                        logger.debug(f"Arquivo carregado com delimitador espaço: {file_path}")
                        return df
                    except Exception as e:
                        logger.error(f"Falha ao carregar {file_path}: {str(e)}")
                        return None
    
    def preprocess_dataset(self, df):
        """
        Pré-processa o dataset para treinamento do modelo.
        
        Args:
            df (DataFrame): DataFrame pandas com os dados brutos
            
        Returns:
            tuple: (X_scaled, y, target_col, task_type) - features escaladas, alvo, coluna alvo e tipo de tarefa
        """
        if df is None or df.empty:
            return None, None, None, None

        # Remove linhas com valores ausentes
        df = df.dropna()
        
        if df.empty:
            return None, None, None, None

        # Identifica colunas numéricas e categóricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        logger.debug(f"Colunas numéricas: {len(numeric_cols)}, Colunas categóricas: {len(categorical_cols)}")
        
        # Codifica características categóricas
        le = LabelEncoder()
        for col in categorical_cols:
            if df[col].nunique() < 100:  # Codifica apenas se não houver muitos valores únicos
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                df = df.drop(columns=[col])
                logger.debug(f"Coluna removida (muitos valores únicos): {col}")
        
        # Tenta identificar a coluna alvo (última coluna ou coluna com 'target', 'class', 'label', etc.)
        potential_targets = ['target', 'class', 'label', 'y', 'outcome', 'diagnosis', 'status']
        target_col = None
        
        # Verifica se algum nome de coluna corresponde a nomes alvo potenciais
        for col in df.columns:
            if col.lower() in potential_targets:
                target_col = col
                break
        
        # Se nenhuma correspondência for encontrada, usa a última coluna
        if target_col is None:
            target_col = df.columns[0]
        
        logger.debug(f"Coluna alvo identificada: {target_col}")
        
        # Features e alvo
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Padroniza as features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
       
        if len(y.unique()) < 10 or pd.api.types.is_categorical_dtype(y):
            task_type = 'classificação'
            # Codifica o alvo se for categórico
            if not pd.api.types.is_numeric_dtype(y):
                y = le.fit_transform(y.astype(str))
        else:
            task_type = 'classificação'
        
        logger.debug(f"Tipo de tarefa identificado: {task_type}")
        return X_scaled, y, target_col, task_type
        
   