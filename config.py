from dataset import ctxcovid
import os


class Config:
    """Configurações centrais do projeto.

    Todos os hiperparâmetros e caminhos são definidos aqui para
    facilitar ajustes e garantir consistência entre os módulos.
    """

    NUM_CLASSES = 3              # Número de classes: Normal, Pneumonia, COVID-19
    BATCH_SIZE = 32              # Tamanho do lote para treino e validação
    LEARNING_RATE = 0.001        # Taxa de aprendizado inicial
    EPOCHS_PER_STAGE = 5         # Épocas por fase de descongelamento gradual
    MAX_UNFREEZE_STAGE = 4       # Número máximo de fases de descongelamento
    MAX_EPOCHS = 25              # Número máximo de épocas de treino
    SEED = 42                    # Seed para reprodutibilidade

    # Caminhos do dataset (retornados pelo KaggleHub)
    BASE_PATH = ctxcovid
    IMAGES_DIR = os.path.join(BASE_PATH, '3A_images')