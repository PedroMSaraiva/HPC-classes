"""
Módulo para seleção de instâncias usando Algoritmo Genético (PyGAD)
"""
import numpy as np
import pygad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
from tqdm import tqdm
from .model import RandomForestModel

class InstanceSelector:
    """
    Classe para seleção de instâncias usando Algoritmo Genético
    """
    
    def __init__(self, num_generations=50, population_size=20, num_parents_mating=4):
        """
        Inicializa o seletor de instâncias
        
        Args:
            num_generations (int): Número de gerações do AG
            population_size (int): Tamanho da população
            num_parents_mating (int): Número de pais para cruzamento
        """
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents_mating = num_parents_mating
        self.rf_model = RandomForestModel()
        self.pbar = None
        self.best_fitness = float('-inf')
        self.generations_without_improvement = 0
        
    def on_generation(self, ga_instance):
        """
        Callback chamado a cada geração do AG
        """
        if self.pbar is not None:
            self.pbar.update(1)
            
        generation = ga_instance.generations_completed
        best_solution = ga_instance.best_solution()
        current_fitness = best_solution[1]
        
        # Atualiza melhor fitness
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.generations_without_improvement = 0
            logger.info(f"Geração {generation}: Novo melhor fitness = {current_fitness:.4f}")
        else:
            self.generations_without_improvement += 1
            
        # Log a cada 10 gerações
        if generation % 10 == 0:
            logger.debug(f"Geração {generation}: Melhor fitness = {current_fitness:.4f}")
            
    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Função de fitness que maximiza acurácia e minimiza número de instâncias
        
        Args:
            ga_instance: Instância do GA
            solution: Solução atual (cromossomo)
            solution_idx: Índice da solução
            
        Returns:
            float: Valor de fitness
        """
        try:
            # Seleciona instâncias baseado na solução binária
            selected_indices = np.where(solution == 1)[0]
            
            if len(selected_indices) == 0:
                return 0.0
            
            # Seleciona subconjunto do dataset usando iloc para indexação posicional
            X_selected = self.X.iloc[selected_indices]
            y_selected = self.y.iloc[selected_indices]
            
            # Divide dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_selected, test_size=0.2, random_state=42
            )
            
            # Treina e avalia o modelo
            self.rf_model.create_model('classificação')
            self.rf_model.train(X_train, y_train)
            y_pred = self.rf_model.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Penaliza soluções com muitas instâncias
            # Ajuste alpha para balancear acurácia vs. redução de instâncias
            alpha = 0.7
            reduction_rate = 1 - (len(selected_indices) / len(self.X))
            
            fitness = (alpha * accuracy) + ((1 - alpha) * reduction_rate)
            
            return fitness
            
        except Exception as e:
            logger.error(f"Erro no cálculo do fitness: {str(e)}")
            return 0.0
    
    def select_instances(self, X, y):
        """
        Seleciona as melhores instâncias usando AG
        
        Args:
            X: Features do dataset
            y: Target do dataset
            
        Returns:
            tuple: (X_selected, y_selected, indices_selected, fitness_history)
        """
        try:
            logger.info("Iniciando seleção de instâncias com AG")
            logger.info(f"Configuração: {self.num_generations} gerações, população={self.population_size}")
            
            self.X = X
            self.y = y
            
            # Configura o AG
            num_genes = len(X)
            
            ga_instance = pygad.GA(
                num_generations=self.num_generations,
                num_parents_mating=self.num_parents_mating,
                num_genes=num_genes,
                init_range_low=0,
                init_range_high=2,
                gene_type=int,
                fitness_func=self.fitness_func,
                sol_per_pop=self.population_size,
                gene_space=[0, 1],
                keep_parents=1,
                crossover_type="uniform",
                mutation_type="random",
                mutation_percent_genes=10,
                on_generation=self.on_generation
            )
            
            # Executa o AG com barra de progresso
            logger.info("Executando algoritmo genético...")
            with tqdm(total=self.num_generations, desc="Progresso do AG") as self.pbar:
                ga_instance.run()
            
            # Obtém a melhor solução
            solution, solution_fitness, _ = ga_instance.best_solution()
            logger.info(f"Melhor fitness encontrado: {solution_fitness:.4f}")
            
            # Seleciona instâncias da melhor solução
            selected_indices = np.where(solution == 1)[0]
            X_selected = X.iloc[selected_indices]
            y_selected = y.iloc[selected_indices]
            
            reduction_rate = (1 - len(X_selected) / len(X)) * 100
            logger.info(f"Redução do dataset: {reduction_rate:.2f}%")
            logger.info(f"Instâncias originais: {len(X)}, Instâncias selecionadas: {len(X_selected)}")
            
            return X_selected, y_selected
            
        except Exception as e:
            logger.error(f"Erro na seleção de instâncias: {str(e)}")
            return None, None, None, None 