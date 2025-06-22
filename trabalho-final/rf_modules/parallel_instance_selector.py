# VERSÃO FINAL COM JOBLIB E CHUNKING - Lógica do Worker Corrigida

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from loguru import logger
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed

# =================================================================================
# FUNÇÃO DE FITNESS GLOBAL (100% AUTÔNOMA)
# =================================================================================
def evaluate_chunk_worker(chunk_de_solucoes, X_data, y_data):
    """
    Função de trabalho que recebe um LOTE (chunk) de soluções e as avalia.
    """
    fitness_do_chunk = []
    for solucao in chunk_de_solucoes:
        try:
            indices_selecionados = np.where(solucao == 1)[0]
            if len(indices_selecionados) < 20:
                fitness_do_chunk.append(0.0)
                continue

            X_subset = X_data.iloc[indices_selecionados]
            y_subset = y_data.iloc[indices_selecionados]

            if len(np.unique(y_subset)) < 2:
                fitness_do_chunk.append(0.0)
                continue

            # CRÍTICO: n_jobs=1 para evitar deadlock
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            
  
            # Treina em uma amostra do SUBSET, não do dataset inteiro
            train_size = min(1000, int(0.75 * len(X_subset)))
            if train_size < 2:
                fitness_do_chunk.append(0.0)
                continue
                
            try:
                X_train, _, y_train, _ = train_test_split(X_subset, y_subset, train_size=train_size, stratify=y_subset, random_state=42)
            except ValueError:
                # Fallback se a estratificação falhar em um subset pequeno
                X_train, _, y_train, _ = train_test_split(X_subset, y_subset, train_size=train_size, random_state=42)
            
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_subset, model.predict(X_subset))
            taxa_reducao = 1.0 - (len(indices_selecionados) / len(X_data))
            
            fitness_do_chunk.append((0.7 * acc) + (0.3 * taxa_reducao))
        except Exception:
            fitness_do_chunk.append(0.0)
    return fitness_do_chunk

class ParallelInstanceSelector:
    """Orquestra a seleção de instâncias em paralelo usando a arquitetura Joblib + Chunking."""
    def __init__(self, num_generations=50, population_size=20, num_parents_mating=4, mutation_percent_genes=10):
        self.n_geracoes, self.tam_pop, self.n_pais, self.taxa_mutacao = num_generations, population_size, num_parents_mating, mutation_percent_genes / 100.0
        self.num_workers = psutil.cpu_count(logical=True)

    def select_instances(self, X, y):
        try:
            logger.info(f"Iniciando seleção com AG Paralelo (Joblib+Chunking) em {self.num_workers} núcleos.")
            n_instancias = X.shape[0]

            populacao = np.random.randint(0, 2, size=(self.tam_pop, n_instancias), dtype=np.int8)
            melhor_fitness_geral, melhor_solucao_geral = -1, populacao[0].copy()

            iterator = tqdm(range(self.n_geracoes), desc="Progresso AG (CPU Joblib)")
            for gen in iterator:
                chunks = np.array_split(populacao, self.num_workers)

                resultados_em_chunks = Parallel(n_jobs=self.num_workers)(
                    delayed(evaluate_chunk_worker)(chunk, X, y) for chunk in chunks
                )
                
                fitness_valores = [item for sublist in resultados_em_chunks for item in sublist]
                fitness_valores = np.array(fitness_valores)

                melhor_fitness_da_geracao = np.max(fitness_valores)
                if melhor_fitness_da_geracao > melhor_fitness_geral:
                    melhor_fitness_geral = melhor_fitness_da_geracao
                    melhor_solucao_geral = populacao[np.argmax(fitness_valores)].copy()

                iterator.set_postfix({'Melhor Fitness': f"{melhor_fitness_geral:.4f}"})
                
                indices_pais = np.argsort(fitness_valores)[-self.n_pais:]
                pais = populacao[indices_pais]
                
                n_filhos = self.tam_pop - self.n_pais
                filhos = np.empty((n_filhos, n_instancias), dtype=np.int8)
                
                for i in range(n_filhos):
                    parentes_crossover = pais[np.random.choice(self.n_pais, 2, replace=False)]
                    ponto_corte = np.random.randint(1, n_instancias - 1) if n_instancias > 1 else 0
                    filho = np.concatenate((parentes_crossover[0, :ponto_corte], parentes_crossover[1, ponto_corte:]))
                    
                    indices_mutacao = np.random.choice(n_instancias, int(n_instancias * self.taxa_mutacao), replace=False)
                    filho[indices_mutacao] = 1 - filho[indices_mutacao]
                    filhos[i, :] = filho
                    
                populacao[0:self.n_pais, :] = pais
                populacao[self.n_pais:, :] = filhos

            logger.success(f"Seleção Paralela concluída. Melhor fitness: {melhor_fitness_geral:.4f}")
            
            indices_finais = np.where(melhor_solucao_geral == 1)[0]
            return X.iloc[indices_finais], y.iloc[indices_finais]
            
        except Exception as e:
            logger.exception(f"Erro fatal na seleção de instâncias paralela: {e}")
            return None, None