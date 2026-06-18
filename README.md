# SIATCT — Sistema Inteligente de Análise de Tomografia Computadorizada de Tórax

Classificador de imagens de **Tomografia Computadorizada (CT)** de tórax utilizando Deep Learning para diagnóstico automatizado de **COVID-19**, **Pneumonia** e casos **Normais**.

O projeto utiliza **PyTorch Lightning** com transfer learning a partir de uma **DenseNet161** pré-treinada no ImageNet, aplicando descongelamento gradual da backbone e técnicas de pré-processamento como CLAHE para realce de contraste em imagens médicas.

---

## Tecnologias

| Tecnologia | Uso |
|------------|-----|
| **Python 3** | Linguagem principal |
| **PyTorch** & **PyTorch Lightning** | Construção e treinamento do modelo |
| **DenseNet161** | Backbone com transfer learning |
| **OpenCV** & **Pillow** | Pré-processamento de imagens (CLAHE, resize, conversão) |
| **scikit-learn** | Métricas de avaliação (ROC, Precision-Recall, Confusion Matrix) |
| **Matplotlib** & **Seaborn** | Geração de gráficos e visualizações |
| **Docker** & **Docker Compose** | Execução reprodutível em container com GPU NVIDIA |

---

## Estrutura do Projeto

```
SIATCT/
├── config.py            # Configurações centrais (hiperparâmetros, caminhos)
├── dataset.py           # Dataset personalizado com CLAHE e transforms
├── loaders.py           # DataLoaders de treino, validação e teste
├── model.py             # DenseNet161 com descongelamento gradual
├── callbacks.py         # Callback de descongelamento gradual por época
├── train.py             # Script principal de treinamento
├── evaluate.py          # Avaliação completa (métricas, gráficos, curvas)
├── visualize.py         # Grad-CAM e Grad-CAM++ — mapas de calor de atenção do modelo
├── calibration.py       # Análise de calibração (ECE) e Temperature Scaling
├── tta.py               # Inferência robusta usando Test-Time Augmentation
├── requirements.txt     # Dependências Python
├── Dockerfile           # Imagem Docker com PyTorch + CUDA
├── docker-compose.yml   # Orquestração do container com GPU
└── README.md            # Este arquivo
```

---

## Arquitetura do Modelo

O classificador utiliza a **DenseNet161** com as seguintes modificações:

1. **Backbone congelada**: todos os pesos pré-treinados do ImageNet são inicialmente congelados.
2. **Classificador personalizado**: o head original é substituído por:
   ```
   Dropout(0.3) -> Linear(2208, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 3)
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

## Avaliação e Métricas

O script `evaluate.py` gera automaticamente:

- **Classification Report** — Precision, Recall e F1-Score por classe
- **Matriz de Confusão** — absoluta e normalizada
- **Curvas de Treinamento** — Loss e Accuracy por época
- **Curva ROC** — AUC por classe (One vs Rest)
- **Curva Precision-Recall** — Average Precision por classe

Outros scripts de análise:

- **`visualize.py`**: Gera mapas de calor **Grad-CAM e Grad-CAM++**, mostrando onde o modelo foca para tomar decisões.
- **`calibration.py`**: Analisa o Expected Calibration Error (ECE), gera o Diagrama de Confiabilidade e aplica **Temperature Scaling** para calibrar as confianças do modelo.
- **`tta.py`**: Compara as métricas de predição normais contra predições consolidadas através de **Test-Time Augmentation (TTA)**.

---

## Como Rodar

### Opção 1: Docker (Recomendado)

#### Pré-requisitos
- Docker e Docker Compose instalados
- GPU NVIDIA com drivers atualizados
- NVIDIA Container Toolkit configurado

#### Configuração do volume do dataset

O dataset deve estar presente localmente. No `docker-compose.yml`, ajuste o caminho local onde o dataset está salvo:
```yaml
volumes:
  - D:\Dataset:/app/data    # Altere para o caminho correto na sua máquina
  - ./outputs:/app/outputs  # Onde os gráficos gerados serão salvos
```

#### Execução

```bash
# Treinar o modelo
docker-compose up --build

# Avaliar o modelo (após treinamento)
docker-compose run classificador python evaluate.py

# Gerar mapas de atenção Grad-CAM
docker-compose run classificador python visualize.py

# Analisar calibração e Temperature Scaling
docker-compose run classificador python calibration.py

# Analisar o impacto do Test-Time Augmentation
docker-compose run classificador python tta.py
```

### Opção 2: Ambiente Local

Certifique-se de configurar a variável `DATASET_PATH` apontando para a pasta raiz do dataset antes de executar.

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate       # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar caminho do dataset (apenas um exemplo)
set DATASET_PATH=D:\Dataset  # Windows
export DATASET_PATH=/caminho/para/dataset # Linux/Mac

# Treinar
python train.py

# Rodar análises
python evaluate.py
python visualize.py
python calibration.py
python tta.py
```

---

## Hiperparâmetros

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

## Otimizações e Técnicas

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: imagens pré-processadas para intensificar bordas em tecidos pulmonares de maneira controlada.
2. **Descongelamento Gradual com LR diferenciado**: learning rates progressivamente menores para camadas mais antigas, garantindo estabilidade no fine-tuning.
3. **Label Smoothing**: a função de perda usa `label_smoothing=0.1` para evitar overfitting.
4. **Precisão Mista (FP16)**: treinamento utilizando `precision="16-mixed"` para reduzir o uso de VRAM.
5. **Calibração (Temperature Scaling)**: ajuste pós-treinamento otimizando a métrica de ECE (Expected Calibration Error).
6. **Test-Time Augmentation (TTA)**: consolidação de inferências através de múltiplas transformações, garantindo robustez na predição em tempo de teste.

---

## Dataset

O projeto utiliza o dataset COVIDx CT-3A, que contém imagens de tomografia computadorizada de tórax classificadas em 3 categorias:

| Classe | Descrição |
|--------|-----------|
| **Normal** | Sem achados patológicos |
| **Pneumonia** | Pneumonia não-COVID |
| **COVID-19** | Pneumonia causada por SARS-CoV-2 |

O projeto espera que o dataset esteja previamente baixado e acessível no diretório apontado pela variável de ambiente `DATASET_PATH`.

---

## Saídas Geradas

As métricas são salvas automaticamente na pasta de saídas (por padrão, `./outputs` localmente ou via docker volume):

| Arquivo | Script | Descrição |
|---------|--------|-----------|
| `checkpoints/best_model.ckpt` | `train.py` | Melhor modelo salvo durante o treino |
| `lightning_csv_logs/` | `train.py` | Métricas de treino/validação por época |
| `confusion_matrix.png` | `evaluate.py` | Matriz de confusão (absoluta) |
| `confusion_matrix_normalized.png` | `evaluate.py` | Matriz de confusão (normalizada) |
| `training_curves.png` | `evaluate.py` | Curvas de Loss e Accuracy |
| `roc_curve.png` | `evaluate.py` | Curva ROC com AUC por classe |
| `precision_recall_curve.png` | `evaluate.py` | Curva Precision-Recall |
| `gradcam_grid.png` | `visualize.py` | Grid de Grad-CAM básico por classe |
| `gradcam_plusplus_grid.png` | `visualize.py` | Grid de Grad-CAM++ por classe |
| `reliability_diagram_before.png` | `calibration.py` | Diagrama de confiabilidade original (ECE bruto) |
| `reliability_diagram_after.png` | `calibration.py` | Diagrama de confiabilidade após Temperature Scaling |
| `tta_comparison.png` | `tta.py` | Comparativo das matrizes de confusão Normal vs TTA |
