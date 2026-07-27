from dataset import ctxcovid
import os


class Config:
    """Configurações centrais do projeto.

    Todos os hiperparâmetros e caminhos são definidos aqui para
    facilitar ajustes e garantir consistência entre os módulos.
    """

    # ── Identificação do experimento ──
    EXPERIMENT_NAME = "original_filtered"   # Mude para cada experimento

    # ── Hiperparâmetros ──
    NUM_CLASSES = 3              # Número de classes: Normal, Pneumonia, COVID-19
    BATCH_SIZE = 64              # Tamanho do lote para treino e validação
    LEARNING_RATE = 5e-4         # Taxa de aprendizado inicial
    MAX_EPOCHS = 40              # Épocas máximas
    SEED = 42                    # Seed para reprodutibilidade
    WEIGHT_DECAY = 1e-4          # Regularização L2

    # ── Gradual Unfreezing ──
    USE_GRADUAL_UNFREEZING = True   # True = descongelamento gradual ativado
    EPOCHS_PER_STAGE = 4            # Épocas por fase de descongelamento gradual
    MAX_UNFREEZE_STAGE = 4          # Número máximo de fases de descongelamento

    # ── Caminhos do dataset (via variável de ambiente DATASET_PATH) ──
    BASE_PATH = ctxcovid

    # ── Diretórios de imagens (separados para suportar segmentação) ──
    IMAGES_DIR_TRAIN = os.path.join(BASE_PATH, '3A_images')        # Imagens de treino/val
    IMAGES_DIR_TEST = os.path.join(BASE_PATH, '3A_images')         # Imagens de teste

    # ── Arquivos de anotação ──
    TRAIN_TXT = os.path.join(BASE_PATH, 'train_filtered.txt')
    VAL_TXT = os.path.join(BASE_PATH, 'val_filtered.txt')
    TEST_TXT = os.path.join(BASE_PATH, 'test_filtered.txt')

    # ── Caminhos de outputs (separados por experimento) ──
    IMG_OUTPUTS_PATH = os.path.join(os.getcwd(), 'outputs', EXPERIMENT_NAME)
    CHECKPOINTS_DIR = os.path.join(BASE_PATH, 'checkpoints', EXPERIMENT_NAME)
    LOGS_DIR = os.path.join(BASE_PATH, 'lightning_csv_logs', EXPERIMENT_NAME)

    @classmethod
    def get_latest_checkpoint(cls):
        """Retorna o caminho para o checkpoint mais recente gerado pelo ModelCheckpoint."""
        from pathlib import Path
        checkpoints_dir = Path(cls.CHECKPOINTS_DIR)
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Diretório de checkpoints não encontrado: {checkpoints_dir}")
        
        # Pega todos os .ckpt, ordenados por data de modificação
        checkpoints = sorted(checkpoints_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            raise FileNotFoundError(f"Nenhum arquivo .ckpt encontrado em {checkpoints_dir}")
        
        return str(checkpoints[-1])
