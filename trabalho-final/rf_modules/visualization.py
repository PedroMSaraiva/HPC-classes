"""
Módulo para visualização de dados e resultados
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import shutil
from loguru import logger
import traceback

# Configuração de estilo para plots
plt.style.use('seaborn-v0_8-darkgrid')  # Estilo mais moderno
sns.set_palette("viridis")  # Paleta de cores moderna

class Visualizer:
    """
    Classe para visualização de resultados de modelos Random Forest
    """
    
    def __init__(self, plots_dir="results/plots"):
        """
        Inicializa o visualizador
        
        Args:
            plots_dir (str): Diretório base para salvar os gráficos
        """
        try:
            self.plots_base_dir = Path(plots_dir).resolve()
            self.plots_base_dir.mkdir(exist_ok=True, parents=True)
            
            # Cria um diretório com timestamp para esta execução
            self.timestamp = datetime.now()
            timestamp_str = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
            self.run_dir = self.plots_base_dir / timestamp_str
            
            logger.info(f"Inicializando visualizador no diretório: {self.plots_base_dir}")
            
            # Cria estrutura de diretórios
            self._setup_directory_structure()
            
            # Configura o estilo global dos gráficos
            self._setup_plot_style()
        except Exception as e:
            logger.error(f"Erro na inicialização do visualizador: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            raise

    def _ensure_directory(self, path):
        """
        Garante que um diretório existe e é acessível
        
        Args:
            path (Path): Caminho do diretório
            
        Returns:
            Path: Caminho absoluto do diretório criado
        """
        try:
            dir_path = Path(path).resolve()
            dir_path.mkdir(exist_ok=True, parents=True)
            logger.debug(f"Diretório garantido: {dir_path}")
            return dir_path
        except Exception as e:
            logger.error(f"Erro ao criar/acessar diretório {path}: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            raise

    def _safe_path(self, base_dir, filename):
        """
        Cria um caminho seguro para salvar arquivos
        
        Args:
            base_dir (Path): Diretório base
            filename (str): Nome do arquivo
            
        Returns:
            Path: Caminho seguro para o arquivo
        """
        try:
            # Garante que o diretório base existe
            base_dir = Path(base_dir).resolve()
            base_dir.mkdir(exist_ok=True, parents=True)
            
            # Cria um caminho seguro usando Path
            safe_path = base_dir / filename
            
            # Garante que o diretório pai do arquivo existe
            safe_path.parent.mkdir(exist_ok=True, parents=True)
            
            logger.debug(f"Caminho seguro criado: {safe_path}")
            return safe_path
        except Exception as e:
            logger.error(f"Erro ao criar caminho seguro para {filename} em {base_dir}: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            raise

    def _save_plot(self, plt, paths, dpi=300):
        """
        Salva um plot em múltiplos caminhos
        
        Args:
            plt: Plot do matplotlib
            paths (list): Lista de caminhos para salvar
            dpi (int): Resolução da imagem
        """
        try:
            for path in paths:
                # Converte barras invertidas para barras normais
                path_str = str(path).replace('\\', '/')
                
                # Garante que o diretório existe
                os.makedirs(os.path.dirname(path_str), exist_ok=True)
                
                # Salva o plot
                plt.savefig(path_str, dpi=dpi)
                logger.debug(f"Plot salvo em: {path_str}")
        except Exception as e:
            logger.error(f"Erro ao salvar plot: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            raise

    def _setup_directory_structure(self):
        """Configura a estrutura de diretórios para os gráficos"""
        
        logger.debug("Configurando estrutura de diretórios para visualizações")
        
        # Cria diretório principal para esta execução
        self.run_dir.mkdir(exist_ok=True)
        
        # Cria subdiretórios para diferentes tipos de gráficos
        self.model_eval_dir = self.run_dir / "model_evaluation"
        self.model_eval_dir.mkdir(exist_ok=True)
        
        self.feature_imp_dir = self.run_dir / "feature_importance"
        self.feature_imp_dir.mkdir(exist_ok=True)
        
        self.comparative_dir = self.run_dir / "comparative_analysis"
        self.comparative_dir.mkdir(exist_ok=True)
        
        self.dataset_dir_mapping = {}  # Mapeamento de datasets para seus diretórios
        
        # Cria um arquivo index.html para fácil navegação
        self._create_html_index()
        
        logger.debug("Estrutura de diretórios configurada com sucesso")
        
    def _create_html_index(self):
        """Cria um arquivo index.html para navegação fácil dos gráficos"""
        logger.debug("Criando arquivo index.html principal")
        index_path = self.plots_base_dir / "index.html"
        
        # Lista de execuções anteriores (pastas com timestamp)
        run_dirs = [d for d in self.plots_base_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
        run_dirs.sort(reverse=True)  # Ordena por mais recentes primeiro
        
        logger.debug(f"Encontradas {len(run_dirs)} execuções anteriores")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Análise Random Forest - Visualizações</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .run-container {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .run-container:hover {{ background-color: #f5f5f5; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Análise Random Forest - Visualizações</h1>
    <p>Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Execuções</h2>
""")
            
            # Adiciona links para cada execução
            for run_dir in run_dirs:
                run_time = datetime.strptime(run_dir.name, "%Y-%m-%d_%H-%M-%S")
                run_time_str = run_time.strftime("%d/%m/%Y às %H:%M:%S")
                
                f.write(f"""
    <div class="run-container">
        <h3><a href="{run_dir.name}/index.html">{run_time_str}</a></h3>
        <p class="timestamp">Diretório: {run_dir.name}</p>
        <ul>
            <li><a href="{run_dir.name}/model_evaluation/index.html">Avaliação de Modelos</a></li>
            <li><a href="{run_dir.name}/feature_importance/index.html">Importância de Features</a></li>
            <li><a href="{run_dir.name}/comparative_analysis/index.html">Análise Comparativa</a></li>
        </ul>
    </div>
""")
            
            f.write("""
</body>
</html>
""")
        
        logger.debug(f"Arquivo index.html principal criado em {index_path}")
    
    def _create_dir_index(self, dir_path, title, description):
        """Cria um arquivo index.html para um diretório específico"""
        logger.debug(f"Criando arquivo index.html para diretório: {dir_path}")
        index_path = dir_path / "index.html"
        
        # Lista todos os arquivos de imagem no diretório
        image_files = [f for f in dir_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        image_files.sort()
        
        logger.debug(f"Encontradas {len(image_files)} imagens no diretório")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .image-container {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        img {{ max-width: 100%; height: auto; margin-top: 10px; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        .back-link {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="back-link">
        <a href="../index.html">&larr; Voltar</a>
    </div>
    
    <h1>{title}</h1>
    <p>{description}</p>
    <p class="timestamp">Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Visualizações</h2>
""")
            
            # Adiciona todas as imagens
            for img_file in image_files:
                # Extrai o nome amigável sem a extensão e timestamp
                img_name = img_file.stem
                img_name_clean = img_name.split('_')[0] if '_' in img_name else img_name
                
                f.write(f"""
    <div class="image-container">
        <h3>{img_name_clean}</h3>
        <p class="timestamp">Arquivo: {img_file.name}</p>
        <a href="{img_file.name}" target="_blank">
            <img src="{img_file.name}" alt="{img_name_clean}">
        </a>
    </div>
""")
            
            f.write("""
</body>
</html>
""")
        
        logger.debug(f"Arquivo index.html criado em {index_path}")
            
    def _update_run_index(self):
        """Atualiza o arquivo index.html do diretório da execução atual"""
        logger.debug("Atualizando arquivo index.html da execução atual")
        index_path = self.run_dir / "index.html"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Execução {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .section {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .section:hover {{ background-color: #f5f5f5; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Execução {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</h1>
    
    <div class="section">
        <h2>Avaliação de Modelos</h2>
        <p>Visualizações relacionadas à performance dos modelos</p>
        <a href="model_evaluation/index.html">Ver gráficos</a>
    </div>
    
    <div class="section">
        <h2>Importância de Features</h2>
        <p>Análise das features mais relevantes para cada modelo</p>
        <a href="feature_importance/index.html">Ver gráficos</a>
    </div>
    
    <div class="section">
        <h2>Análise Comparativa</h2>
        <p>Comparação entre diferentes datasets e modelos</p>
        <a href="comparative_analysis/index.html">Ver gráficos</a>
    </div>
</body>
</html>
""")
        
        logger.debug(f"Arquivo index.html da execução atualizado em {index_path}")
        
    def _get_dataset_dir(self, dataset_name):
        """
        Obtém ou cria um diretório para um dataset específico
        
        Args:
            dataset_name (str): Nome do dataset
            
        Returns:
            Path: Caminho para o diretório do dataset
        """
        if dataset_name not in self.dataset_dir_mapping:
            # Pega apenas a primeira parte do nome antes do primeiro '_' ou espaço
            safe_name = dataset_name.split('_')[0].split(' ')[0]
            safe_name = ''.join(c for c in safe_name if c.isalnum())
            
            # Cria o diretório do dataset
            dataset_dir = self.run_dir / "data" / safe_name
            dataset_dir.mkdir(exist_ok=True, parents=True)
            
            self.dataset_dir_mapping[dataset_name] = dataset_dir
            
        return self.dataset_dir_mapping[dataset_name]
        
    def _setup_plot_style(self):
        """Configura o estilo global para os gráficos"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
    
    def _add_timestamp_to_filename(self, filename):
        """
        Adiciona timestamp ao nome do arquivo
        
        Args:
            filename (str): Nome do arquivo
            
        Returns:
            str: Nome do arquivo com timestamp
        """
        name, ext = os.path.splitext(filename)
        # Pega apenas a primeira parte do nome
        name = name.split('_')[0].split(' ')[0]
        name = ''.join(c for c in name if c.isalnum())
        
        # Usa formato mais curto para timestamp
        timestamp_str = datetime.now().strftime("%m%d_%H%M")
        return f"{name}_{timestamp_str}{ext}"
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names, dataset_name, timestamp=None):
        """
        Plota e salva a matriz de confusão para modelos de classificação.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            class_names: Nomes das classes
            dataset_name (str): Nome do dataset para o título
            timestamp: Timestamp do treinamento
        """
        try:
            logger.info(f"Gerando matriz de confusão para {dataset_name}")
            
            # Cria a matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12, 10))
            
            # Cria uma heatmap mais bonita
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=.5, cbar_kws={"shrink": 0.8})
                        
            # Melhora a aparência
            plt.title(f'Matriz de Confusão - {dataset_name}', fontsize=18, pad=20)
            plt.ylabel('Valor Real', fontsize=16, labelpad=15)
            plt.xlabel('Valor Predito', fontsize=16, labelpad=15)
            
            # Adiciona informação de timestamp
            if timestamp:
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                plt.figtext(0.99, 0.01, f"Gerado em: {timestamp_str}", 
                            horizontalalignment='right', size=10, 
                            color='gray', style='italic')
            
            # Salva nos diretórios apropriados
            plt.tight_layout()
            
            # Gera nome de arquivo seguro
            filename = self._add_timestamp_to_filename(f"{dataset_name}_confusion_matrix.png")
            
            # Cria caminhos seguros para salvar
            eval_path = self._safe_path(self.model_eval_dir, filename)
            dataset_dir = self._get_dataset_dir(dataset_name)
            dataset_path = self._safe_path(dataset_dir, filename)
            
            # Salva o plot
            self._save_plot(plt, [eval_path, dataset_path])
            
            logger.info(f"Matriz de confusão salva com sucesso para {dataset_name}")
            
            plt.close()
            
            # Atualiza os arquivos de índice
            self._create_dir_index(self.model_eval_dir, "Avaliação de Modelos", 
                                "Visualizações das métricas de desempenho dos modelos")
            self._create_dir_index(dataset_dir, f"Dataset: {dataset_name}", 
                                f"Visualizações para o dataset {dataset_name}")
            self._update_run_index()
            
        except Exception as e:
            logger.error(f"Erro ao gerar matriz de confusão para {dataset_name}: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            if plt:
                plt.close()
        
    def plot_regression_results(self, y_true, y_pred, dataset_name, timestamp=None):
        """
        Plota e salva os resultados para modelos de regressão.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            dataset_name (str): Nome do dataset para o título
            timestamp: Timestamp do treinamento
        """
        try:
            logger.info(f"Gerando gráfico de regressão para {dataset_name}")
            plt.figure(figsize=(12, 10))
            
            # Adiciona um degradê de cor com base na densidade dos pontos
            cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
            scatter = plt.scatter(y_true, y_pred, alpha=0.6, c=np.abs(y_true - y_pred), cmap=cmap, s=50)
            
            # Adiciona linha de referência
            lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
            plt.plot(lims, lims, 'r--', linewidth=2, label='Predição Perfeita')
            
            # Calcula e adiciona estatísticas
            mse = np.mean((y_true - y_pred) ** 2)
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            plt.annotate(f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
            
            # Adiciona barra de cores para mostrar o erro
            cbar = plt.colorbar(scatter)
            cbar.set_label('|Erro|', rotation=270, labelpad=20)
            
            plt.title(f'Valores Reais vs. Preditos - {dataset_name}', fontsize=18, pad=20)
            plt.xlabel('Valores Reais', fontsize=16, labelpad=15)
            plt.ylabel('Valores Preditos', fontsize=16, labelpad=15)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower right')
            
            # Adiciona informação de timestamp
            if timestamp:
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                plt.figtext(0.99, 0.01, f"Gerado em: {timestamp_str}", 
                            horizontalalignment='right', size=10, 
                            color='gray', style='italic')
            
            # Salva nos diretórios apropriados
            plt.tight_layout()
            
            # Gera nome de arquivo seguro
            filename = self._add_timestamp_to_filename(f"{dataset_name}_regression_results.png")
            
            # Cria caminhos seguros para salvar
            eval_path = self._safe_path(self.model_eval_dir, filename)
            dataset_dir = self._get_dataset_dir(dataset_name)
            dataset_path = self._safe_path(dataset_dir, filename)
            
            # Salva o plot
            self._save_plot(plt, [eval_path, dataset_path])
            
            logger.debug(f"Gráfico de regressão salvo em {eval_path} e {dataset_path}")
            
            plt.close()
            
            # Atualiza os arquivos de índice
            self._create_dir_index(self.model_eval_dir, "Avaliação de Modelos", 
                                "Visualizações das métricas de desempenho dos modelos")
            self._create_dir_index(dataset_dir, f"Dataset: {dataset_name}", 
                                f"Visualizações para o dataset {dataset_name}")
            self._update_run_index()
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de regressão para {dataset_name}: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            if plt:
                plt.close()
        
    def plot_feature_importance(self, feature_importance, dataset_name, top_n=15, timestamp=None):
        """
        Plota e salva a importância das features.
        
        Args:
            feature_importance (DataFrame): DataFrame com importância das features
            dataset_name (str): Nome do dataset para o título
            top_n (int): Número de top features para mostrar
            timestamp: Timestamp do treinamento
        """
        try:
            logger.info(f"Gerando gráfico de importância de features para {dataset_name}")
            plt.figure(figsize=(14, 10))
            
            # Pega as top features
            top_features = feature_importance.head(top_n)
            
            # Cria um gráfico de barras horizontal com degradê de cores
            bars = sns.barplot(x='importance', y='feature', data=top_features, 
                               palette='viridis')
            
            # Adiciona valores à direita das barras
            for i, v in enumerate(top_features['importance']):
                plt.text(v + 0.005, i, f"{v:.4f}", va='center')
            
            plt.title(f'Top {top_n} Features Importantes - {dataset_name}', fontsize=18, pad=20)
            plt.xlabel('Importância', fontsize=16, labelpad=15)
            plt.ylabel('Feature', fontsize=16, labelpad=15)
            
            # Adiciona informação de timestamp
            if timestamp:
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                plt.figtext(0.99, 0.01, f"Gerado em: {timestamp_str}", 
                            horizontalalignment='right', size=10, 
                            color='gray', style='italic')
            
            # Salva nos diretórios apropriados
            plt.tight_layout()
            
            # Gera nome de arquivo seguro
            filename = self._add_timestamp_to_filename(f"{dataset_name}_feature_importance.png")
            
            # Cria caminhos seguros para salvar
            feature_path = self._safe_path(self.feature_imp_dir, filename)
            dataset_dir = self._get_dataset_dir(dataset_name)
            dataset_path = self._safe_path(dataset_dir, filename)
            
            # Salva o plot
            self._save_plot(plt, [feature_path, dataset_path])
            
            logger.debug(f"Gráfico de importância de features salvo em {feature_path} e {dataset_path}")
            
            plt.close()
            
            # Atualiza os arquivos de índice
            self._create_dir_index(self.feature_imp_dir, "Importância de Features", 
                                "Visualizações da importância das features para cada modelo")
            self._create_dir_index(dataset_dir, f"Dataset: {dataset_name}", 
                                f"Visualizações para o dataset {dataset_name}")
            self._update_run_index()
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de importância de features para {dataset_name}: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            if plt:
                plt.close()
        
    def create_comparative_plots(self, results):
        """
        Cria gráficos comparativos com referência temporal para todos os datasets processados.
        
        Args:
            results (dict): Dicionário com resultados para todos os datasets
        """
        logger.info("Iniciando criação de gráficos comparativos")
        
        # Extrai dados para os gráficos
        dataset_names = []
        accuracies = []
        r2_scores = []
        training_times = []
        load_times = []
        preprocess_times = []
        timestamps = []
        task_types = []
        
        for dataset_name, result in results.items():
            dataset_names.append(dataset_name)
            training_times.append(result['metrics'].get('training_time', 0))
            load_times.append(result['metrics'].get('load_time', 0))
            preprocess_times.append(result['metrics'].get('preprocess_time', 0))
            timestamps.append(result['metrics'].get('timestamp', pd.Timestamp.now()))
            task_types.append(result['task_type'])
            
            if result['task_type'] == 'classificação':
                accuracies.append(result['metrics'].get('accuracy', None))
                r2_scores.append(None)
            else:
                accuracies.append(None)
                r2_scores.append(result['metrics'].get('r2', None))
        
        logger.debug(f"Processando dados comparativos para {len(dataset_names)} datasets")
        
        # Cria DataFrame para facilitar a plotagem
        df_results = pd.DataFrame({
            'dataset': dataset_names,
            'accuracy': accuracies,
            'r2': r2_scores,
            'training_time': training_times,
            'load_time': load_times,
            'preprocess_time': preprocess_times,
            'timestamp': timestamps,
            'task_type': task_types
        })
        
        # Separa resultados por tipo de tarefa
        df_classification = df_results[df_results['task_type'] == 'classificação'].copy()
        df_regression = df_results[df_results['task_type'] == 'regressão'].copy()
        
        logger.debug(f"Datasets de classificação: {len(df_classification)}, Datasets de regressão: {len(df_regression)}")
        
        # Ordena por timestamp para ver evolução temporal
        df_classification = df_classification.sort_values('timestamp')
        df_regression = df_regression.sort_values('timestamp')
        
        # Gráfico de acurácia para classificação com linha temporal
        if not df_classification.empty:
            logger.debug("Gerando gráfico temporal de acurácia para classificação")
            self._plot_temporal_metric(
                df_classification, 
                'accuracy', 
                'Acurácia', 
                'Comparação de Acurácia entre Datasets (Classificação)',
                'classification_accuracy_temporal'
            )
        
        # Gráfico de R² para regressão com linha temporal
        if not df_regression.empty:
            logger.debug("Gerando gráfico temporal de R² para regressão")
            self._plot_temporal_metric(
                df_regression, 
                'r2', 
                'R²', 
                'Comparação de R² entre Datasets (Regressão)',
                'regression_r2_temporal'
            )
        
        # Gráfico de tempo de treinamento
        logger.debug("Gerando gráfico temporal de tempo de treinamento")
        self._plot_temporal_metric(
            df_results, 
            'training_time', 
            'Tempo de Treinamento (s)', 
            'Comparação de Tempo de Treinamento entre Datasets',
            'training_time_temporal'
        )
        
        # Gráfico comparativo dos diferentes tempos
        logger.debug("Gerando gráfico comparativo de tempos")
        self._plot_time_comparison(df_results)
        
        # Atualiza os arquivos de índice
        self._create_dir_index(self.comparative_dir, "Análise Comparativa", 
                              "Visualizações comparativas entre datasets")
        self._update_run_index()
        
        logger.success("Gráficos comparativos gerados com sucesso")
    
    def _plot_temporal_metric(self, df, metric_col, metric_name, title, filename):
        """
        Cria gráfico temporal de uma métrica.
        
        Args:
            df (DataFrame): DataFrame com os dados
            metric_col (str): Nome da coluna de métrica
            metric_name (str): Nome legível da métrica para o gráfico
            title (str): Título do gráfico
            filename (str): Nome do arquivo para salvar
        """
        try:
            logger.debug(f"Gerando gráfico temporal para {metric_name}")
            plt.figure(figsize=(16, 10))
            
            # Configurar cores
            palette = sns.color_palette("viridis", n_colors=len(df))
            
            # Criar barplot
            barplot = sns.barplot(
                x='dataset', 
                y=metric_col, 
                data=df,
                palette=palette
            )
            
            # Rotaciona nomes dos datasets
            plt.xticks(rotation=45, ha='right')
            
            # Adiciona linha temporal
            ax2 = plt.twinx()
            timestamps = mdates.date2num(df['timestamp'])
            ax2.plot(range(len(df)), timestamps, 'r-', marker='o', linewidth=2, markersize=8)
            ax2.set_ylabel('Data/Hora', color='r', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Formatação da data
            date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
            ax2.yaxis.set_major_formatter(date_format)
            
            # Adiciona valores às barras
            for i, v in enumerate(df[metric_col]):
                plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            
            plt.title(title, fontsize=18, pad=20)
            plt.xlabel('Dataset', fontsize=16, labelpad=15)
            plt.ylabel(metric_name, fontsize=16, labelpad=15)
            
            # Adiciona informação sobre quando o gráfico foi gerado
            plt.figtext(0.99, 0.01, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                        horizontalalignment='right', size=10, 
                        color='gray', style='italic')
            
            plt.tight_layout()
            
            # Salva com timestamp no nome do arquivo
            timestamped_filename = self._add_timestamp_to_filename(f"{filename}.png")
            save_path = self.comparative_dir / timestamped_filename
            plt.savefig(save_path, dpi=300)
            logger.debug(f"Gráfico temporal salvo em {save_path}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico temporal para {metric_name}: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            plt.close()
    
    def _plot_time_comparison(self, df):
        """
        Cria um gráfico comparativo dos diferentes tempos (carregamento, pré-processamento, treinamento)
        por dataset.
        
        Args:
            df (DataFrame): DataFrame com os dados de tempo
        """
        if len(df) == 0:
            logger.warning("Nenhum dado disponível para gerar gráfico de comparação de tempos")
            return
            
        try:
            logger.debug("Gerando gráfico de comparação de tempos")
            # Prepara dados para gráfico de barras empilhadas
            plt.figure(figsize=(16, 10))
            
            # Ordena por tempo total (soma dos tempos)
            df['total_time'] = df['load_time'] + df['preprocess_time'] + df['training_time']
            df_sorted = df.sort_values('total_time', ascending=False)
            
            # Cria o gráfico de barras empilhadas
            bottom_vals = np.zeros(len(df_sorted))
            
            # Define cores mais distintas
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            labels = ['Carregamento', 'Pré-processamento', 'Treinamento']
            
            # Adiciona cada tipo de tempo como uma camada de barras
            for i, col in enumerate(['load_time', 'preprocess_time', 'training_time']):
                plt.bar(df_sorted['dataset'], df_sorted[col], bottom=bottom_vals, 
                        label=labels[i], color=colors[i], alpha=0.8)
                bottom_vals += df_sorted[col].values
            
            # Rotaciona nomes dos datasets
            plt.xticks(rotation=45, ha='right')
            
            # Adiciona valores no topo de cada barra para tempo total
            for i, (dataset, total) in enumerate(zip(df_sorted['dataset'], df_sorted['total_time'])):
                plt.text(i, total + 0.05, f"{total:.2f}s", ha='center', va='bottom', fontweight='bold')
            
            # Formata o gráfico
            plt.title('Comparação dos Tempos de Processamento por Dataset', fontsize=18, pad=20)
            plt.xlabel('Dataset', fontsize=16, labelpad=15)
            plt.ylabel('Tempo (segundos)', fontsize=16, labelpad=15)
            plt.legend(title='Fase', title_fontsize=14, fontsize=12, loc='upper right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adiciona timestamp de geração
            plt.figtext(0.99, 0.01, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                        horizontalalignment='right', size=10, color='gray', style='italic')
            
            plt.tight_layout()
            
            # Salva com timestamp no nome do arquivo
            timestamped_filename = self._add_timestamp_to_filename("time_comparison.png")
            save_path = self.comparative_dir / timestamped_filename
            plt.savefig(save_path, dpi=300)
            logger.debug(f"Gráfico de comparação de tempos salvo em {save_path}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de comparação de tempos: {str(e)}")
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            plt.close()
        
    def finalize(self):
        """
        Finaliza a geração de visualizações e atualiza todos os índices HTML
        """
        logger.info("Finalizando geração de visualizações e atualizando índices")
        
        # Atualiza os HTML para cada diretório
        self._create_dir_index(self.model_eval_dir, "Avaliação de Modelos", 
                             "Visualizações das métricas de desempenho dos modelos")
        self._create_dir_index(self.feature_imp_dir, "Importância de Features", 
                             "Visualizações da importância das features para cada modelo")
        self._create_dir_index(self.comparative_dir, "Análise Comparativa", 
                             "Visualizações comparativas entre datasets")
        
        # Atualiza o index da execução
        self._update_run_index()
        
        # Atualiza o index principal
        self._create_html_index()
        
        logger.success(f"Visualizações organizadas em: {self.run_dir}")
        logger.success(f"Acesse os resultados em HTML: {self.plots_base_dir}/index.html")

    def plot_fitness_evolution(self, fitness_history, dataset_name, save_dir):
        """
        Plota a evolução do fitness durante a seleção de instâncias
        
        Args:
            fitness_history (array): Histórico de fitness
            dataset_name (str): Nome do dataset
            save_dir (Path): Diretório para salvar o gráfico
        """
        try:
            plt.figure(figsize=(12, 8))
            generations = range(1, len(fitness_history) + 1)
            
            plt.plot(generations, fitness_history, 'b-', linewidth=2)
            plt.scatter(generations, fitness_history, c='blue', alpha=0.5)
            
            plt.title(f'Evolução do Fitness - {dataset_name}', fontsize=16)
            plt.xlabel('Geração', fontsize=14)
            plt.ylabel('Fitness', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Adiciona informações sobre melhor fitness
            best_fitness = max(fitness_history)
            best_gen = fitness_history.index(best_fitness) + 1
            plt.axhline(y=best_fitness, color='r', linestyle='--', alpha=0.5)
            plt.text(0.02, 0.98, f'Melhor Fitness: {best_fitness:.4f}\nGeração: {best_gen}',
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Salva o gráfico
            save_path = Path(save_dir) / f"fitness_evolution_{dataset_name.replace('/', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Gráfico de evolução do fitness salvo em {save_path}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar evolução do fitness: {str(e)}")
            if plt:
                plt.close()
    
    def plot_parallel_fitness_evolution(self, fitness_history, dataset_name, save_dir):
        """
        Plota a evolução do fitness para múltiplas execuções paralelas
        
        Args:
            fitness_history (list): Lista com os valores de fitness de cada execução
            dataset_name (str): Nome do dataset
            save_dir (Path): Diretório para salvar o gráfico
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plota o fitness de cada execução
            for i, fitness in enumerate(fitness_history):
                plt.scatter(i+1, fitness, c='blue', alpha=0.6, s=100)
            
            plt.axhline(y=max(fitness_history), color='r', linestyle='--', 
                       label=f'Melhor Fitness: {max(fitness_history):.4f}')
            
            plt.title(f'Fitness por Execução Paralela - {dataset_name}', fontsize=16)
            plt.xlabel('Execução', fontsize=14)
            plt.ylabel('Fitness', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Salva o gráfico
            save_path = save_dir / f"parallel_fitness_{self._safe_filename(dataset_name)}.png"
            self._save_plot(plt, [save_path])
            plt.close()
            
        except Exception as e:
            logger.error(f"Erro ao plotar evolução do fitness paralelo: {str(e)}")
            
    def plot_accuracy_comparison(self, results, dataset_name, save_dir):
        """
        Plota comparação entre acurácias original e com seleção
        
        Args:
            results (dict): Dicionário com os resultados
            dataset_name (str): Nome do dataset
            save_dir (Path): Diretório para salvar o gráfico
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Dados para o gráfico
            accuracies = [results['original_accuracy'], results['selected_accuracy']]
            labels = ['Dataset Original', 'Dataset Reduzido']
            
            # Cria barras
            bars = plt.bar(labels, accuracies, color=['lightblue', 'lightgreen'])
            
            # Adiciona valores sobre as barras
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')
            
            plt.title(f'Comparação de Acurácia - {dataset_name}', fontsize=16)
            plt.ylabel('Acurácia', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Adiciona informação sobre redução
            plt.figtext(0.02, 0.02, 
                       f'Redução de instâncias: {results["reduction_rate"]:.2f}%',
                       fontsize=10)
            
            # Salva o gráfico
            save_path = save_dir / f"accuracy_comparison_{self._safe_filename(dataset_name)}.png"
            self._save_plot(plt, [save_path])
            plt.close()
            
        except Exception as e:
            logger.error(f"Erro ao plotar comparação de acurácias: {str(e)}")
            
    def plot_overall_comparison(self, results_df, save_dir):
        """
        Plota comparação geral entre todos os datasets
        
        Args:
            results_df (DataFrame): DataFrame com os resultados
            save_dir (Path): Diretório para salvar o gráfico
        """
        try:
            plt.figure(figsize=(15, 10))
            
            # Prepara dados para o gráfico
            datasets = results_df['dataset_name']
            x = np.arange(len(datasets))
            width = 0.35
            
            # Cria barras
            plt.bar(x - width/2, results_df['original_accuracy'], 
                   width, label='Original', color='lightblue')
            plt.bar(x + width/2, results_df['selected_accuracy'], 
                   width, label='Reduzido', color='lightgreen')
            
            plt.title('Comparação de Acurácia por Dataset', fontsize=16)
            plt.xlabel('Dataset', fontsize=14)
            plt.ylabel('Acurácia', fontsize=14)
            plt.xticks(x, datasets, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Ajusta layout
            plt.tight_layout()
            
            # Salva o gráfico
            save_path = save_dir / "overall_comparison.png"
            self._save_plot(plt, [save_path])
            plt.close()
            
        except Exception as e:
            logger.error(f"Erro ao plotar comparação geral: {str(e)}") 