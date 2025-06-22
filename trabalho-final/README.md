# Análise de Seleção de Instâncias com Algoritmos Genéticos

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![Status: Ativo](https://img.shields.io/badge/Status-Ativo-success.svg)](https://github.com/)

Este projeto implementa e compara algoritmos sequenciais e paralelos para seleção de instâncias usando Algoritmos Genéticos, com o objetivo de reduzir o tamanho de datasets mantendo ou melhorando a qualidade da classificação.

## 📋 Visão Geral

O projeto utiliza Algoritmos Genéticos para selecionar subconjuntos otimizados de instâncias de datasets, buscando:
- **Reduzir o tamanho dos datasets** (menos instâncias)
- **Manter ou melhorar a acurácia** dos modelos de classificação
- **Acelerar o treinamento** de modelos futuros
- **Comparar performance** entre implementações sequenciais e paralelas

## 🚀 Funcionalidades

### Algoritmos Implementados
- **Seleção Sequencial**: Processamento tradicional dataset por dataset
- **Seleção Paralela**: Processamento simultâneo usando múltiplos cores
- **Comparação Automática**: Análise comparativa de performance
- **Validação de Qualidade**: Verificação da qualidade dos datasets reduzidos

### Métricas Coletadas
- Tempo de execução
- Taxa de redução de instâncias
- Fitness do algoritmo genético
- Speedup paralelo
- Acurácia de classificação
- Métricas de qualidade (precision, recall, F1-score)

## 📁 Estrutura do Projeto

```
trabalho-final/
├── dataset/                          # Datasets originais (arquivos CSV)
├── rf_modules/                       # Módulos principais
│   ├── instance_selector.py         # Classe do seletor de instâncias
│   └── logger_setup.py              # Configuração de logging
├── results/                          # Resultados gerados
│   ├── instance_selection/          # Resultados sequenciais
│   ├── parallel_instance_selection/ # Resultados paralelos
│   ├── reduced_datasets/            # Datasets reduzidos
│   ├── comparison_plots/            # Gráficos comparativos
│   ├── comparison_reports/          # Relatórios de comparação
│   └── validation/                  # Validação de qualidade
├── run_instance_selection.py        # Script sequencial
├── run_parallel_instance_selection.py # Script paralelo
├── compare_algorithms.py            # Comparação de algoritmos
├── validate_reduced_datasets.py     # Validação de qualidade
├── run_complete_analysis.py         # Pipeline completo
├── requirements.txt                 # Dependências
└── README.md                        # Este arquivo
```

## 🔧 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação das Dependências
```bash
pip install -r requirements.txt
```

### Dependências Principais
- `numpy`: Computação numérica
- `pandas`: Manipulação de dados
- `scikit-learn`: Algoritmos de machine learning
- `matplotlib` + `seaborn`: Visualização
- `pygad`: Algoritmos genéticos
- `psutil`: Monitoramento de sistema

## 📊 Uso

### Preparação dos Dados
1. Coloque seus datasets CSV no diretório `dataset/`
2. **Formato esperado**: Primeira coluna = classe/target, demais colunas = features
3. Certifique-se de que os dados estão limpos e preprocessados

### Execução Completa (Recomendado)
```bash
python run_complete_analysis.py
```

Este comando executa todo o pipeline:
1. Seleção sequencial de instâncias
2. Seleção paralela de instâncias  
3. Comparação de algoritmos
4. Validação da qualidade dos datasets reduzidos

### Execução Individual

#### Seleção Sequencial
```bash
python run_instance_selection.py
```

#### Seleção Paralela
```bash
python run_parallel_instance_selection.py
```

#### Comparação de Algoritmos
```bash
python compare_algorithms.py
```

#### Validação de Qualidade
```bash
python validate_reduced_datasets.py
```

## 📈 Interpretação dos Resultados

### Métricas de Performance
- **Execution Time**: Tempo de execução em segundos
- **Reduction Rate**: Percentual de instâncias removidas
- **Best Fitness**: Qualidade da solução (0-1, maior é melhor)
- **Speedup**: Aceleração obtida com paralelização
- **Accuracy Change**: Mudança na acurácia após redução

### Critérios de Qualidade
- **Bom resultado**: Alta redução (>30%) + Acurácia mantida/melhorada
- **Resultado aceitável**: Redução moderada (15-30%) + Pequena perda de acurácia (<2%)
- **Resultado problemático**: Baixa redução (<15%) ou grande perda de acurácia (>5%)

### Arquivos de Saída

#### Datasets Reduzidos
- `results/reduced_datasets/reduced_[nome_dataset].csv`
- Formato: primeira coluna = target, demais = features selecionadas

#### Métricas de Performance
- `results/instance_selection/sequential_performance_metrics_[timestamp].csv`
- `results/parallel_instance_selection/parallel_performance_metrics_[timestamp].csv`

#### Relatórios de Comparação
- `results/comparison_reports/algorithm_comparison_[timestamp].csv`
- `results/comparison_reports/detailed_results_[timestamp].csv`

#### Validação de Qualidade
- `results/validation/dataset_comparisons_[timestamp].csv`
- `results/validation/validation_summary_[timestamp].csv`

#### Visualizações
- `results/comparison_plots/algorithm_comparison.png`
- `results/comparison_plots/detailed_comparison.png`

## ⚙️ Configuração dos Algoritmos

### Parâmetros do Algoritmo Genético
Os parâmetros podem ser ajustados em `rf_modules/instance_selector.py`:

```python
# Configurações padrão
num_generations = 50        # Número de gerações
population_size = 100       # Tamanho da população
num_parents_mating = 50     # Número de pais para reprodução
mutation_probability = 0.1  # Probabilidade de mutação
alpha = 0.7                # Peso da acurácia vs redução (0.7 = 70% acurácia, 30% redução)
```

### Configuração Paralela
O número de workers paralelos é automaticamente detectado baseado no número de cores da CPU, mas pode ser ajustado em `run_parallel_instance_selection.py`.

## 🔍 Monitoramento e Logs

O sistema gera logs detalhados durante a execução:
- Progresso do algoritmo genético
- Métricas de cada dataset processado
- Tempos de execução
- Erros e avisos

Os logs são exibidos no console e podem ser redirecionados para arquivos se necessário.

## 🎯 Exemplos de Uso

### Exemplo 1: Análise Básica
```bash
# Coloque seus CSVs em dataset/
python run_complete_analysis.py
# Verifique os resultados em results/
```

### Exemplo 2: Comparação Rápida
```bash
# Execute apenas os algoritmos
python run_instance_selection.py
python run_parallel_instance_selection.py
# Compare os resultados
python compare_algorithms.py
```

### Exemplo 3: Validação Específica
```bash
# Após ter datasets reduzidos
python validate_reduced_datasets.py
```

## 📊 Métricas de Avaliação

### Performance Computacional
- **Tempo de execução**: Duração total do processamento
- **Speedup**: Aceleração obtida com paralelização
- **Eficiência**: Percentual de utilização dos recursos paralelos
- **Instâncias por segundo**: Taxa de processamento

### Qualidade dos Resultados
- **Taxa de redução**: Percentual de instâncias removidas
- **Acurácia**: Precisão do modelo de classificação
- **F1-Score**: Média harmônica entre precision e recall
- **Fitness**: Função objetivo do algoritmo genético

## 🔬 Algoritmo Genético

### Representação
- **Cromossomo**: Vetor binário representando seleção de instâncias
- **Gene**: Bit indicando se uma instância é selecionada (1) ou não (0)

### Função de Fitness
```
fitness = α × accuracy + (1-α) × reduction_rate
```
Onde:
- `α = 0.7`: Peso da acurácia (70%)
- `1-α = 0.3`: Peso da redução (30%)

### Operadores Genéticos
- **Seleção**: Torneio baseado em fitness
- **Cruzamento**: Cruzamento uniforme
- **Mutação**: Flip de bits com probabilidade controlada

## 🚨 Solução de Problemas

### Erro: "Diretório 'dataset' não encontrado"
- Certifique-se de que existe um diretório `dataset/` no diretório de trabalho
- Coloque pelo menos um arquivo CSV no diretório

### Erro: "Módulo não encontrado"
- Execute `pip install -r requirements.txt`
- Verifique se está usando o Python correto (3.8+)

### Performance Lenta
- Reduza o número de gerações ou tamanho da população
- Use datasets menores para testes iniciais
- Verifique se há recursos suficientes de CPU/RAM

### Resultados Inconsistentes
- O algoritmo genético é estocástico - execute múltiplas vezes
- Ajuste a semente aleatória para reprodutibilidade
- Aumente o número de gerações para convergência

## 📝 Notas Importantes

1. **Formato dos Dados**: Os CSVs devem ter a classe/target na primeira coluna
2. **Normalização**: Os dados são automaticamente normalizados
3. **Validação Cruzada**: Utiliza 5-fold stratified cross-validation
4. **Reprodutibilidade**: Seeds fixas garantem resultados consistentes
5. **Recursos**: Algoritmos paralelos utilizam todos os cores disponíveis

## 🤝 Contribuição

Para contribuir com o projeto:
1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Implemente as mudanças
4. Teste thoroughly
5. Submeta um pull request

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e de pesquisa.

---

**Desenvolvido para análise de seleção de instâncias com foco em performance e qualidade.**
