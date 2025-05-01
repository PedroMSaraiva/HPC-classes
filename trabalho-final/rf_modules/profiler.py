"""
Módulo para análise e visualização de desempenho
"""
import time
import cProfile
import pstats
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from functools import wraps
from datetime import datetime
import numpy as np
import multiprocessing

class Profiler:
    """
    Classe para análise de desempenho e profiling
    """
    
    def __init__(self, results_dir="results/profiling"):
        """
        Inicializa o profiler
        
        Args:
            results_dir (str): Diretório para salvar resultados de profiling
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.timings = {}
        self.start_times = {}
        self.profiles = {}
        
    def profile_function(self, func_name=None):
        """
        Decorador para profile de função
        
        Args:
            func_name (str, opcional): Nome da função para registro (se None, usa o nome real)
            
        Returns:
            callable: Decorador de função
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                function_name = func_name if func_name else func.__name__
                
                # Verifica se está em um processo principal (não em um processo filho)
                # para evitar conflitos de profiling em ambientes multiprocessados
                is_main_process = multiprocessing.current_process().name == 'MainProcess'
                
                # Apenas usa cProfile no processo principal
                profiler = None
                if is_main_process:
                    try:
                        profiler = cProfile.Profile()
                        profiler.enable()
                    except ValueError:
                        # Se já existe um profiler ativo, apenas continua sem profiling
                        profiler = None
                
                # Registra início
                start_time = time.time()
                self.start_times[function_name] = start_time
                
                # Executa função
                result = func(*args, **kwargs)
                
                # Registra término
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Desativa o profiler se ele foi ativado
                if profiler:
                    profiler.disable()
                    
                    # Armazena resultados
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats(20)  # top 20 funções
                    self.profiles[function_name] = s.getvalue()
                
                # Registra os tempos em todos os casos
                if function_name not in self.timings:
                    self.timings[function_name] = []
                self.timings[function_name].append({
                    'execution_time': execution_time,
                    'timestamp': datetime.now(),
                    'args': str(args),
                })
                
                return result
            return wrapper
        return decorator
    
    def time_function_start(self, function_name):
        """
        Inicia a medição de tempo para uma função
        
        Args:
            function_name (str): Nome da função
        """
        self.start_times[function_name] = time.time()
    
    def time_function_end(self, function_name, args_info=None):
        """
        Finaliza a medição de tempo para uma função
        
        Args:
            function_name (str): Nome da função
            args_info (str, opcional): Informação adicional sobre argumentos
        """
        if function_name not in self.start_times:
            return
            
        end_time = time.time()
        execution_time = end_time - self.start_times[function_name]
        
        if function_name not in self.timings:
            self.timings[function_name] = []
            
        self.timings[function_name].append({
            'execution_time': execution_time,
            'timestamp': datetime.now(),
            'args': args_info if args_info else '',
        })
    
    def save_profile_results(self, filename_prefix="profile"):
        """
        Salva os resultados de profiling em arquivos
        
        Args:
            filename_prefix (str): Prefixo para nomes de arquivos
        """
        # Salva perfis em texto
        with open(self.results_dir / f"{filename_prefix}_details.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RESULTADOS DETALHADOS DE PROFILING\n")
            f.write("=" * 80 + "\n\n")
            
            for func_name, profile in self.profiles.items():
                f.write(f"FUNÇÃO: {func_name}\n")
                f.write("-" * 80 + "\n")
                f.write(profile)
                f.write("\n\n")
        
        # Cria dataframe para análise
        profile_data = []
        for func_name, timing_list in self.timings.items():
            for timing in timing_list:
                profile_data.append({
                    'function': func_name,
                    'execution_time': timing['execution_time'],
                    'timestamp': timing['timestamp'],
                    'args': timing['args']
                })
        
        if not profile_data:
            return
            
        df_profile = pd.DataFrame(profile_data)
        df_profile.to_csv(self.results_dir / f"{filename_prefix}_summary.csv", index=False)
        
        # Cria visualizações
        self._create_profile_visualizations(df_profile, filename_prefix)
    
    def _create_profile_visualizations(self, df_profile, filename_prefix):
        """
        Cria visualizações dos dados de profiling
        
        Args:
            df_profile (DataFrame): DataFrame com os dados de profiling
            filename_prefix (str): Prefixo para nomes de arquivos
        """
        # Configuração de estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")
        
        # Gráfico de barras de tempo médio por função
        plt.figure(figsize=(14, 8))
        df_means = df_profile.groupby('function')['execution_time'].mean().reset_index().sort_values('execution_time', ascending=False)
        
        ax = sns.barplot(x='execution_time', y='function', data=df_means, palette='viridis')
        
        # Adiciona os valores nas barras
        for i, v in enumerate(df_means['execution_time']):
            ax.text(v + 0.1, i, f"{v:.4f}s", va='center')
        
        plt.title('Tempo Médio de Execução por Função', fontsize=16)
        plt.xlabel('Tempo (segundos)', fontsize=14)
        plt.ylabel('Função', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{filename_prefix}_mean_times.png", dpi=300)
        plt.close()
        
        # Gráfico de linha temporal
        if 'timestamp' in df_profile.columns and len(df_profile['function'].unique()) > 1:
            plt.figure(figsize=(16, 10))
            
            pivot_df = df_profile.pivot_table(
                index='timestamp', 
                columns='function', 
                values='execution_time', 
                aggfunc='mean'
            ).reset_index()
            
            pivot_df = pivot_df.set_index('timestamp')
            
            # Plota linha por função
            ax = pivot_df.plot(marker='o', linewidth=2, markersize=8, figsize=(16, 10))
            
            plt.title('Tempo de Execução ao Longo do Tempo', fontsize=16)
            plt.xlabel('Timestamp', fontsize=14)
            plt.ylabel('Tempo (segundos)', fontsize=14)
            plt.legend(title='Função', title_fontsize=14, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / f"{filename_prefix}_time_series.png", dpi=300)
            plt.close() 