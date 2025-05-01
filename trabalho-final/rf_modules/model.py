"""
Módulo para o modelo Random Forest e cálculos relacionados
"""
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from numba import jit
from loguru import logger

class RandomForestModel:
    """
    Classe para criação, treinamento e avaliação de modelos Random Forest
    """
    
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        """
        Inicializa o modelo
        
        Args:
            n_estimators (int): Número de árvores na floresta
            random_state (int): Semente aleatória
            n_jobs (int): Número de jobs para paralelização (-1 para usar todos os cores)
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.task_type = None
        self.training_time = None
        self.timestamp = None
    
    def create_model(self, task_type):
        """
        Cria o modelo apropriado com base no tipo de tarefa
        
        Args:
            task_type (str): Tipo de tarefa ('classificação' ou 'regressão')
            
        Returns:
            object: Modelo criado
        """
        self.task_type = task_type
        
        if task_type == 'classificação':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators, 
                random_state=self.random_state, 
                n_jobs=self.n_jobs
            )
            self.cv_scoring = 'accuracy'
        else:  # regressão
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                random_state=self.random_state, 
                n_jobs=self.n_jobs
            )
            self.cv_scoring = 'neg_mean_squared_error'
        
        return self.model
    
    def train(self, X_train, y_train):
        """
        Treina o modelo e registra o tempo de treinamento
        
        Args:
            X_train: Features de treinamento
            y_train: Alvo de treinamento
            
        Returns:
            object: Modelo treinado
        """
        if self.model is None:
            raise ValueError("O modelo precisa ser criado antes do treinamento.")
            
        # Registra o timestamp de início
        self.timestamp = pd.Timestamp.now()
            
        # Treina e mede o tempo do modelo
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        logger.debug(f"Modelo treinado em {self.training_time:.4f} segundos")
        return self.model
    
    def predict(self, X_test):
        """
        Faz previsões com o modelo
        
        Args:
            X_test: Features de teste
            
        Returns:
            array: Previsões
        """
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de fazer previsões.")
            
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test, X=None, y=None):
        """
        Avalia o modelo com diferentes métricas
        
        Args:
            X_test: Features de teste
            y_test: Alvo de teste
            X: Todas as features (para validação cruzada)
            y: Todos os alvos (para validação cruzada)
            
        Returns:
            dict: Métricas de desempenho
        """
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de ser avaliado.")
            
        metrics = {}
        metrics['training_time'] = self.training_time
        metrics['timestamp'] = self.timestamp
        
        # Faz previsões
        y_pred = self.predict(X_test)
        
        # Calcula pontuação de validação cruzada se X e y forem fornecidos
        if X is not None and y is not None:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring=self.cv_scoring)
            metrics['cv_scores_mean'] = np.mean(cv_scores)
            metrics['cv_scores_std'] = np.std(cv_scores)
        
        # Métricas específicas para cada tipo de tarefa
        if self.task_type == 'classificação':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['classification_report'] = classification_report(y_test, y_pred)
        else:  # regressão
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
        
        return metrics, y_pred
        
    @staticmethod
    @jit(forceobj=True)
    def calculate_feature_importance(model, X):
        """
        Calcula a importância das features usando Numba para aceleração.
        
        Args:
            model: Modelo treinado de Random Forest
            X (DataFrame): DataFrame com as features
            
        Returns:
            DataFrame: DataFrame com as importâncias das features ordenadas
        """
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance 