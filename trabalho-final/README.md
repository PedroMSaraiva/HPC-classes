# 🔬 Sistema de Seleção de Instâncias com Algoritmo Genético e Random Forest

&#x20;  &#x20;

Este projeto expande o sistema modular original de análise com Random Forest ao incluir **seleção de instâncias** com **Algoritmo Genético (AG)** e **execução paralela com Joblib**. O objetivo é reduzir o número de instâncias nos datasets mantendo (ou até melhorando) a performance preditiva dos modelos.

---

## 🧐 Novas Funcionalidades

- Seleção de instâncias com algoritmo genético (AG tradicional)
- Versão paralela com **Joblib + Chunking**
- Avaliação de desempenho com Random Forest e KNN
- Métricas de redução de dados e acurácia combinadas
- Processamento multiprocessado com aproveitamento total da CPU

---

## 📂 Estrutura do Projeto

```
trabalho-final/
├── run_instance_selection.py              # Versão com AG sequencial
├── run_parallel_instance_selection.py     # Versão com AG paralelo (Joblib)
├── rf_modules/
│   ├── instance_selector.py               # Seleção de instâncias com PyGAD
│   ├── parallel_instance_selector.py      # Seleção paralela com Joblib
│   ├── model.py                           # RandomForestModel e abstração de treino
│   └── ...                                # Outros módulos (data_loader, logger, etc.)
├── docs                                   # Documentação
├── dataset/                               # Datasets de entrada
├── results/
│   ├── instance_selection/                # Resultados do AG tradicional
│   ├── parallel_instance_selection/       # Resultados da versão paralela
│   └── logs/                              # Arquivos de log e profiling
```

---

## ⚙️ Scripts Disponíveis

### ▶️ `run_instance_selection.py`

Executa seleção de instâncias com **PyGAD** para problemas de classificação:

- Otimiza uma máscara binária que define quais linhas do dataset manter
- Avalia fitness com Random Forest, equilibrando acurácia e redução de dados
- Salva os dados reduzidos em `results/instance_selection/`

---

### 🥵 `run_parallel_instance_selection.py`

Executa a versão **paralela** com **Joblib** e KNN para avaliação mais leve:

- Divide a população de AG em chunks e avalia em paralelo
- Usa `KNeighborsClassifier` para acelerar fitness
- Estratégia robusta de fallback para subsets pequenos

---

### 🧬 `instance_selector.py`

Módulo com `InstanceSelector`, baseado em **PyGAD**, que:

- Treina modelos Random Forest a cada geração
- Calcula fitness balanceando acurácia e taxa de redução
- Mostra progresso com `tqdm` e logs detalhados

---

### 🧠 `parallel_instance_selector.py`

Módulo com `ParallelInstanceSelector` que:

- Avalia chunks da população com `joblib.Parallel`
- Usa `KNN` com avaliação rápida para escalabilidade
- Aplica crossover e mutação manuais a cada geração

---

## 📊 Métrica de Fitness

Ambos os métodos utilizam a seguinte fórmula:

```
fitness = (α × acurácia) + ((1 - α) × taxa_reducao)
```

- Valor padrão de `α = 0.7`
- Penaliza soluções que mantêm muitas instâncias
- Evita overfitting em subconjuntos muito pequenos

---

## 🚀 Como Executar

### Ambiente Conda

Crie e ative o ambiente com o arquivo `environment.yml` fornecido:

```bash
conda env create -f environment.yml
conda activate rf-env
```

### Execução

#### Versão Genética Simples (PyGAD)

```bash
python run_instance_selection.py
```

#### Versão Paralela (Joblib + Chunking)

```bash
python run_parallel_instance_selection.py
```

---

## 💾 Resultados

- Os datasets reduzidos serão salvos em `results/instance_selection/` ou `results/parallel_instance_selection/`
- Os arquivos CSV contêm apenas as instâncias selecionadas com suas respectivas labels
- Logs completos de execução estão disponíveis no console e/ou `logs/`

---

## 🧽 Próximos Passos

- Suporte à seleção de **atributos (features)** com algoritmos evolutivos
- Integração com interface HTML para visualização das soluções
- Avaliação com múltiplos modelos: SVM, XGBoost, LightGBM
- Multiobjetivo: acurácia, redução, tempo e generalização

---

## 📝 Licença

Este projeto está licenciado sob a **Licença MIT** – consulte o arquivo `LICENSE`.

