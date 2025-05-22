"""
Módulo para seleção de instâncias usando Algoritmo Genético com Paralelização
"""
import numpy as np
import pygad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .model import RandomForestModel
import joblib
from functools import partial
import psutil
import warnings

# Tenta importar diferentes backends de GPU
GPU_BACKEND = None
try:
    import torch
    if torch.cuda.is_available():
        GPU_BACKEND = 'pytorch'
        logger.info("Usando PyTorch com CUDA para aceleração GPU")
except ImportError:
    pass

if GPU_BACKEND is None:
    try:
        import cupy as cp
        GPU_BACKEND = 'cupy'
        logger.info("Usando CuPy para aceleração GPU")
    except ImportError:
        pass

if GPU_BACKEND is None:
    try:
        from numba import cuda
        if cuda.is_available():
            GPU_BACKEND = 'numba'
            logger.info("Usando Numba CUDA para aceleração GPU")
    except ImportError:
        pass

if GPU_BACKEND is None:
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            GPU_BACKEND = 'tensorflow'
            logger.info("Usando TensorFlow para aceleração GPU")
    except ImportError:
        pass

class ParallelInstanceSelector:
    """
    Classe para seleção de instâncias usando AG com avaliação paralela de indivíduos
    """
    
    def __init__(self, 
                 num_generations=50, 
                 population_size=20, 
                 num_parents_mating=4,
                 mutation_percent_genes=10,
                 init_fitness_cache=True,
                 use_gpu=True):
        """
        Inicializa o seletor de instâncias
        
        Args:
            num_generations (int): Número de gerações do AG
            population_size (int): Tamanho da população
            num_parents_mating (int): Número de pais para cruzamento
            mutation_percent_genes (int): Porcentagem de genes para mutação
            init_fitness_cache (bool): Se deve inicializar cache de fitness
            use_gpu (bool): Se deve usar GPU para aceleração
        """
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents_mating = num_parents_mating
        self.mutation_percent_genes = mutation_percent_genes
        self.rf_model = RandomForestModel()
        self.fitness_cache = {} if init_fitness_cache else None
        self.best_fitness = float('-inf')
        self.generations_without_improvement = 0
        self.early_stopping_generations = 10
        self.min_selected_ratio = 0.1  # Mínimo de 10% das instâncias
        self.max_selected_ratio = 0.9  # Máximo de 90% das instâncias
        
        # Configuração de GPU/CPU
        self.use_gpu = use_gpu and GPU_BACKEND is not None
        self.gpu_backend = GPU_BACKEND if self.use_gpu else None
        
        # Usa todos os cores lógicos disponíveis
        if self.use_gpu:
            if self.gpu_backend == 'pytorch':
                self.device = torch.device('cuda')
                self.num_workers = torch.cuda.device_count()
            elif self.gpu_backend == 'cupy':
                self.num_workers = cp.cuda.runtime.getDeviceCount()
            elif self.gpu_backend == 'numba':
                self.num_workers = len(cuda.list_devices())
            elif self.gpu_backend == 'tensorflow':
                self.num_workers = len(tf.config.list_physical_devices('GPU'))
            logger.info(f"Usando {self.num_workers} GPU(s) com backend {self.gpu_backend}")
        else:
            self.num_workers = psutil.cpu_count(logical=True)  # Usa todos os cores lógicos
            logger.info(f"Usando {self.num_workers} CPU cores para paralelização")
            
        # Pre-compila o modelo para cada worker
        self.rf_models = [RandomForestModel() for _ in range(self.num_workers)]
        for model in self.rf_models:
            model.create_model('classificação')
    
    def _to_gpu(self, data):
        """
        Transfere dados para GPU usando o backend apropriado
        """
        if not self.use_gpu:
            return data
            
        try:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.values
                
            if self.gpu_backend == 'pytorch':
                return torch.tensor(data, device=self.device)
            elif self.gpu_backend == 'cupy':
                return cp.array(data)
            elif self.gpu_backend == 'tensorflow':
                return tf.convert_to_tensor(data, device='/GPU:0')
            else:
                return data  # Para Numba, mantém em numpy
        except Exception as e:
            logger.warning(f"Erro ao transferir para GPU: {str(e)}")
            return data
    
    def _to_cpu(self, data):
        """
        Transfere dados de volta para CPU
        """
        if not self.use_gpu:
            return data
            
        try:
            if self.gpu_backend == 'pytorch':
                return data.cpu().numpy()
            elif self.gpu_backend == 'cupy':
                return cp.asnumpy(data)
            elif self.gpu_backend == 'tensorflow':
                return data.numpy()
            else:
                return data  # Para Numba, já está em numpy
        except Exception as e:
            logger.warning(f"Erro ao transferir para CPU: {str(e)}")
            return data
    
    def _evaluate_individual(self, args):
        """
        Avalia um indivíduo da população
        
        Args:
            args: Tupla (solution, worker_id)
            
        Returns:
            float: Valor de fitness
        """
        try:
            solution, worker_id = args
            
            # Verifica cache de fitness
            if self.fitness_cache is not None:
                solution_key = solution.tobytes()
                if solution_key in self.fitness_cache:
                    return self.fitness_cache[solution_key]
            
            # Garante que solution é um array numpy
            solution = np.asarray(solution)
            if solution.ndim == 0:
                solution = np.atleast_1d(solution)
            
            # Seleciona instâncias usando boolean indexing
            selected_indices = np.arange(len(solution))[solution == 1]
            n_total = len(self.X)
            n_selected = len(selected_indices)
            
            # Verifica proporção de instâncias selecionadas
            selected_ratio = n_selected / n_total
            if selected_ratio < self.min_selected_ratio or selected_ratio > self.max_selected_ratio:
                return 0.1
            
            X_selected = self.X.iloc[selected_indices]
            y_selected = self.y.iloc[selected_indices]
            
            # Garante pelo menos 2 amostras por classe
            unique_classes, class_counts = np.unique(y_selected, return_counts=True)
            if len(unique_classes) < len(np.unique(self.y)) or np.any(class_counts < 2):
                return 0.1
            
            # Usa divisão rápida para datasets pequenos
            if len(X_selected) < 100:
                test_size = 0.2
                indices = np.arange(len(X_selected))
                np.random.shuffle(indices)
                split = int(0.8 * len(indices))
                train_idx, test_idx = indices[:split], indices[split:]
                
                X_train = X_selected.iloc[train_idx]
                X_test = X_selected.iloc[test_idx]
                y_train = y_selected.iloc[train_idx]
                y_test = y_selected.iloc[test_idx]
            else:
                # Para datasets maiores, usa stratified split
                test_size = 0.2
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y_selected,
                        test_size=test_size,
                        stratify=y_selected,
                        random_state=42
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y_selected,
                        test_size=test_size,
                        random_state=42
                    )
            
            # Usa o modelo pré-compilado do worker
            rf_model = self.rf_models[worker_id]
            
            if self.use_gpu:
                # Converte dados para GPU
                X_train_gpu = self._to_gpu(X_train)
                y_train_gpu = self._to_gpu(y_train)
                X_test_gpu = self._to_gpu(X_test)
                y_test_gpu = self._to_gpu(y_test)
                
                # Treina modelo
                rf_model.train(self._to_cpu(X_train_gpu), self._to_cpu(y_train_gpu))
                y_pred = rf_model.model.predict(self._to_cpu(X_test_gpu))
                accuracy = accuracy_score(self._to_cpu(y_test_gpu), y_pred)
            else:
                rf_model.train(X_train, y_train)
                y_pred = rf_model.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
            
            # Calcula fitness otimizado
            reduction_score = 1 - abs(0.5 - selected_ratio)
            class_balance = np.min(class_counts) / np.max(class_counts)
            
            # Pesos ajustados para favorecer acurácia
            fitness = (0.7 * accuracy) + (0.15 * reduction_score) + (0.15 * class_balance)
            fitness = 0.1 + 0.9 * fitness
            
            # Armazena no cache
            if self.fitness_cache is not None:
                self.fitness_cache[solution_key] = fitness
            
            return fitness
            
        except Exception as e:
            logger.error(f"Erro na avaliação do indivíduo: {str(e)}")
            return 0.1
            
    def _parallel_fitness(self, solutions, X, y):
        """
        Avalia fitness de múltiplos indivíduos em paralelo
        
        Args:
            solutions: Lista de soluções
            X: Features do dataset
            y: Target do dataset
            
        Returns:
            list: Lista de valores de fitness
        """
        try:
            # Prepara argumentos com worker_id
            args = [(sol, i % self.num_workers) for i, sol in enumerate(solutions)]
            
            if self.use_gpu:
                if self.gpu_backend == 'numba':
                    # Divide soluções entre as GPUs disponíveis
                    chunks = np.array_split(args, self.num_workers)
                    fitness_values = []
                    
                    for i, chunk in enumerate(chunks):
                        with cuda.gpus[i % self.num_workers]:
                            chunk_fitness = list(map(self._evaluate_individual, chunk))
                            fitness_values.extend(chunk_fitness)
                else:
                    # Outros backends (PyTorch, CuPy, TensorFlow)
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        fitness_values = list(executor.map(self._evaluate_individual, args))
            else:
                # Usa ProcessPoolExecutor para CPU com todos os cores
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    fitness_values = list(executor.map(self._evaluate_individual, args))
            
            return fitness_values
                
        except Exception as e:
            logger.error(f"Erro na avaliação paralela: {str(e)}")
            return [0.1] * len(solutions)
    
    def fitness_func(self, ga_instance, solutions, _):
        """
        Função de fitness que avalia toda a população em paralelo
        
        Args:
            ga_instance: Instância do GA
            solutions: População atual
            _: Índice da solução (não usado)
            
        Returns:
            list: Lista de valores de fitness
        """
        try:
            # Avalia população em paralelo
            fitness_values = self._parallel_fitness(solutions, self.X, self.y)
            
            # Atualiza melhor fitness
            current_best = max(fitness_values)
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.generations_without_improvement = 0
                logger.info(f"Geração {ga_instance.generations_completed}: Novo melhor fitness = {current_best:.4f}")
            else:
                self.generations_without_improvement += 1
                
            # Early stopping
            if self.generations_without_improvement >= self.early_stopping_generations:
                logger.info(f"Early stopping após {self.generations_without_improvement} gerações sem melhoria")
                ga_instance.stop_criteria = ["reach_0"]
            
            # Log a cada 5 gerações
            if ga_instance.generations_completed % 5 == 0:
                logger.debug(f"Geração {ga_instance.generations_completed}: Melhor fitness = {current_best:.4f}")
                logger.debug(f"Distribuição de fitness: min={min(fitness_values):.4f}, max={max(fitness_values):.4f}, mean={np.mean(fitness_values):.4f}")
            
            return fitness_values
            
        except Exception as e:
            logger.error(f"Erro no cálculo do fitness: {str(e)}")
            return [0.0] * len(solutions)
    
    def select_instances(self, X, y):
        """
        Seleciona as melhores instâncias usando AG com paralelização
        
        Args:
            X: Features do dataset
            y: Target do dataset
            
        Returns:
            tuple: (X_selected, y_selected, indices_selected, fitness_history)
        """
        try:
            logger.info("Iniciando seleção de instâncias com AG paralelo")
            logger.info(f"Configuração: {self.num_generations} gerações, população={self.population_size}")
            if self.use_gpu:
                logger.info(f"Usando GPU ({self.gpu_backend}) com {self.num_workers} dispositivo(s)")
            else:
                logger.info(f"Usando CPU com {self.num_workers} cores lógicos")
            
            self.X = X
            self.y = y
            
            # Configura o AG
            num_genes = len(X)
            min_genes = int(num_genes * self.min_selected_ratio)
            max_genes = int(num_genes * self.max_selected_ratio)
            
            # Inicialização inteligente da população
            def init_population():
                population = []
                # Adiciona algumas soluções com distribuição uniforme de genes
                for ratio in np.linspace(0.2, 0.8, self.population_size // 2):
                    n_ones = int(num_genes * ratio)
                    solution = np.zeros(num_genes, dtype=int)
                    ones_indices = np.random.choice(num_genes, size=n_ones, replace=False)
                    solution[ones_indices] = 1
                    population.append(solution)
                
                # Completa com soluções aleatórias
                remaining = self.population_size - len(population)
                for _ in range(remaining):
                    n_ones = np.random.randint(min_genes, max_genes + 1)
                    solution = np.zeros(num_genes, dtype=int)
                    ones_indices = np.random.choice(num_genes, size=n_ones, replace=False)
                    solution[ones_indices] = 1
                    population.append(solution)
                
                return np.array(population)
            
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
                keep_parents=2,  # Aumentado para manter mais boas soluções
                crossover_type="uniform",
                mutation_type="random",
                mutation_percent_genes=self.mutation_percent_genes,
                initial_population=init_population(),
                parallel_processing=["thread", self.num_workers],  # Usa todos os cores disponíveis
                stop_criteria=["reach_0", f"saturate_{self.early_stopping_generations}"]
            )
            
            # Executa o AG com barra de progresso
            logger.info("Executando algoritmo genético...")
            with tqdm(total=self.num_generations, desc="Progresso do AG") as pbar:
                while ga_instance.generations_completed < ga_instance.num_generations:
                    ga_instance.run()  # Executa uma geração por vez
                    pbar.update(1)
                    pbar.set_postfix({'Melhor Fitness': f"{self.best_fitness:.4f}"})
                    
                    # Verifica critério de parada
                    if ga_instance.stop_criteria:
                        logger.info("Critério de parada atingido")
                        break
            
            # Obtém a melhor solução
            solution, solution_fitness, _ = ga_instance.best_solution()
            selected_indices = np.where(solution == 1)[0]
            
            # Seleciona instâncias
            X_selected = X.iloc[selected_indices]
            y_selected = y.iloc[selected_indices]
            
            # Calcula estatísticas
            reduction_rate = (1 - len(X_selected) / len(X)) * 100
            logger.info(f"Melhor fitness encontrado: {solution_fitness:.4f}")
            logger.info(f"Redução do dataset: {reduction_rate:.2f}%")
            logger.info(f"Instâncias originais: {len(X)}, Instâncias selecionadas: {len(X_selected)}")
            
            # Limpa cache e libera memória GPU
            if self.fitness_cache is not None:
                self.fitness_cache.clear()
            
            # Libera memória GPU de acordo com o backend
            if self.use_gpu:
                if self.gpu_backend == 'pytorch':
                    torch.cuda.empty_cache()
                elif self.gpu_backend == 'cupy':
                    cp.get_default_memory_pool().free_all_blocks()
                elif self.gpu_backend == 'tensorflow':
                    tf.keras.backend.clear_session()
            
            return X_selected, y_selected, selected_indices, ga_instance.best_solutions_fitness
            
        except Exception as e:
            logger.error(f"Erro na seleção de instâncias: {str(e)}")
            return None, None, None, None