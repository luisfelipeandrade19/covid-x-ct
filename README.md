# 🫁 SIATCT — Sistema Inteligente de Análise de Tomografia Computadorizada de Tórax

Classificador de imagens de **Tomografia Computadorizada (CT)** de tórax utilizando Deep Learning para diagnóstico automatizado de **COVID-19**, **Pneumonia** e casos **Normais**.

O projeto utiliza **PyTorch Lightning** com transfer learning a partir de uma **DenseNet161** pré-treinada no ImageNet, aplicando descongelamento gradual da backbone e técnicas de pré-processamento como CLAHE para realce de contraste em imagens médicas.

---

## 🚀 Tecnologias

| Tecnologia | Uso |
|------------|-----|
| **Python 3** | Linguagem principal |
| **PyTorch** & **PyTorch Lightning** | Construção e treinamento do modelo |
| **DenseNet161** | Backbone com transfer learning |
| **OpenCV** & **Pillow** | Pré-processamento de imagens (CLAHE, resize, conversão) |
| **scikit-learn** | Métricas de avaliação (ROC, Precision-Recall, Confusion Matrix) |
| **Matplotlib** & **Seaborn** | Geração de gráficos e visualizações |
| **KaggleHub** | Download automático do dataset `hgunraj/covidxct` |
| **Docker** & **Docker Compose** | Execução reprodutível em container com GPU NVIDIA |

---

## 📂 Estrutura do Projeto

```
📁 SIATCT/
├── config.py            # Configurações centrais (hiperparâmetros, caminhos)
├── dataset.py           # Dataset personalizado com CLAHE e transforms
├── loaders.py           # DataLoaders de treino, validação e teste
├── model.py             # DenseNet161 com descongelamento gradual
├── callbacks.py         # Callback de descongelamento gradual por época
├── train.py             # Script principal de treinamento
├── evaluate.py          # Avaliação completa (métricas, gráficos, curvas)
├── visualize.py         # Grad-CAM — mapas de calor de atenção do modelo
├── requirements.txt     # Dependências Python
├── Dockerfile           # Imagem Docker com PyTorch + CUDA
├── docker-compose.yml   # Orquestração do container com GPU
└── README.md            # Este arquivo
```

---

## 🧠 Arquitetura do Modelo

O classificador utiliza a **DenseNet161** com as seguintes modificações:

1. **Backbone congelada**: todos os pesos pré-treinados do ImageNet são inicialmente congelados.
2. **Classificador personalizado**: o head original é substituído por:
   ```
   Dropout(0.3) → Linear(2208, 512) → ReLU → Dropout(0.2) → Linear(512, 3)
   ```
3. **Descongelamento gradual**: a cada N épocas, uma nova camada da backbone é descongelada (do bloco mais profundo ao mais raso), com learning rates diferenciados por camada.

### Fases de descongelamento

| Fase | Blocos descongelados |
|------|---------------------|
| 0 | Nenhum (apenas classificador) |
| 1 | `denseblock4`, `norm5` |
| 2 | `denseblock3`, `transition3` |
| 3 | `denseblock2`, `transition2` |
| 4 | `denseblock1`, `transition1`, `conv0`, `norm0` |

---

## 📊 Avaliação e Métricas

O script `evaluate.py` gera automaticamente:

- **Classification Report** — Precision, Recall e F1-Score por classe
- **Matriz de Confusão** — absoluta e normalizada
- **Curvas de Treinamento** — Loss e Accuracy por época
- **Curva ROC** — AUC por classe (One vs Rest)
- **Curva Precision-Recall** — Average Precision por classe

O script `visualize.py` gera:

- **Grad-CAM** — mapas de calor mostrando onde o modelo foca para tomar decisões, essencial para validação clínica

---

## 🛠 Como Rodar

### Opção 1: Docker (Recomendado)

#### Pré-requisitos
- Docker e Docker Compose instalados
- GPU NVIDIA com drivers atualizados
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) configurado

#### Configuração do volume do dataset

No `docker-compose.yml`, ajuste o caminho local onde o dataset será salvo:
```yaml
volumes:
  - D:\Dataset:/app/data    # Altere para o caminho desejado na sua máquina
```

#### Execução

```bash
# Treinar o modelo
docker-compose up --build

# Avaliar o modelo (após treinamento)
docker-compose run classificador python evaluate.py

# Gerar Grad-CAM (após treinamento)
docker-compose run classificador python visualize.py
```

### Opção 2: Ambiente Local

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate       # Windows

# Instalar dependências
pip install -r requirements.txt

# Treinar
python train.py

# Avaliar
python evaluate.py

# Gerar Grad-CAM
python visualize.py
```

---

## ⚙️ Hiperparâmetros

Todos os hiperparâmetros são configuráveis em `config.py`:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `NUM_CLASSES` | 3 | Normal, Pneumonia, COVID-19 |
| `BATCH_SIZE` | 32 | Tamanho do lote |
| `LEARNING_RATE` | 0.001 | Taxa de aprendizado inicial |
| `EPOCHS_PER_STAGE` | 5 | Épocas entre fases de descongelamento |
| `MAX_UNFREEZE_STAGE` | 4 | Número máximo de fases |
| `MAX_EPOCHS` | 25 | Limite de épocas de treino |

---

## 🔬 Otimizações e Técnicas

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: cada imagem de CT é processada com CLAHE para intensificar bordas e diferenças nos tecidos pulmonares de maneira controlada, melhorando a capacidade do modelo de distinguir padrões sutis.

2. **Descongelamento Gradual com LR diferenciado**: a backbone é descongelada progressivamente do bloco mais profundo ao mais raso. Camadas descongeladas mais cedo recebem learning rates menores, garantindo estabilidade durante o fine-tuning.

3. **Label Smoothing**: a função de perda usa `label_smoothing=0.1` para evitar overfitting e melhorar a calibração das probabilidades.

4. **Precisão Mista (FP16)**: o treinamento utiliza `precision="16-mixed"` para reduzir o uso de VRAM e acelerar o treino sem perda significativa de qualidade.

5. **Early Stopping**: o treinamento é interrompido automaticamente se a `val_loss` não melhorar por 5 épocas consecutivas.

6. **Data Augmentation**: o conjunto de treino aplica flip horizontal, rotação aleatória (±10°) e variação de brilho/contraste para aumentar a diversidade dos dados.

---

## 📁 Dataset

O projeto utiliza o dataset [COVIDx CT-3A](https://www.kaggle.com/datasets/hgunraj/covidxct), que contém imagens de tomografia computadorizada de tórax classificadas em 3 categorias:

| Classe | Descrição |
|--------|-----------|
| **Normal** | Sem achados patológicos |
| **Pneumonia** | Pneumonia não-COVID |
| **COVID-19** | Pneumonia causada por SARS-CoV-2 |

O download é feito automaticamente via `kagglehub` na primeira execução.

---

## 📄 Saídas Geradas

Após a execução completa (treino + avaliação + visualização), os seguintes arquivos são gerados:

| Arquivo | Script | Descrição |
|---------|--------|-----------|
| `checkpoints/best_model.ckpt` | `train.py` | Melhor modelo salvo durante o treino |
| `lightning_csv_logs/` | `train.py` | Métricas de treino/validação por época |
| `confusion_matrix.png` | `evaluate.py` | Matriz de confusão (absoluta) |
| `confusion_matrix_normalized.png` | `evaluate.py` | Matriz de confusão (normalizada) |
| `training_curves.png` | `evaluate.py` | Curvas de Loss e Accuracy |
| `roc_curve.png` | `evaluate.py` | Curva ROC com AUC por classe |
| `precision_recall_curve.png` | `evaluate.py` | Curva Precision-Recall |
| `gradcam_grid.png` | `visualize.py` | Grid de Grad-CAM por classe |
