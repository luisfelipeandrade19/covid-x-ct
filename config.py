from dataset import ctxcovid
import os


class Config:
    """Configurações centrais do projeto.

    Todos os hiperparâmetros e caminhos são definidos aqui para
    facilitar ajustes e garantir consistência entre os módulos.
    """

    NUM_CLASSES = 3              # Número de classes: Normal, Pneumonia, COVID-19
    BATCH_SIZE = 64              # Tamanho do lote para treino e validação
    LEARNING_RATE = 5e-4        # Taxa de aprendizado inicial
    EPOCHS_PER_STAGE = 4         # Épocas por fase de descongelamento gradual
    MAX_UNFREEZE_STAGE = 4       # Número máximo de fases de descongelamento
    MAX_EPOCHS = 25               
    SEED = 42                    # Seed para reprodutibilidade
    WEIGHT_DECAY = 1e-4          # Regularização L2

    # Caminhos do dataset (via variável de ambiente DATASET_PATH)
    BASE_PATH = ctxcovid
    IMAGES_DIR = os.path.join(BASE_PATH, '3A_images')

    # Caminho de outputs
    IMG_OUTPUTS_PATH = os.path.join(os.getcwd(), 'outputs')