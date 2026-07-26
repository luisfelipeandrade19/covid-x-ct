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
    MAX_EPOCHS = 40               # Aumentado para dar tempo ao CBAM de aprender               
    SEED = 42                    # Seed para reprodutibilidade
    WEIGHT_DECAY = 1e-4          # Regularização L2

    # Caminhos do dataset (via variável de ambiente DATASET_PATH)
    BASE_PATH = ctxcovid
    IMAGES_DIR = os.path.join(BASE_PATH, '3A_images')

    # Caminho de outputs
    IMG_OUTPUTS_PATH = os.path.join(os.getcwd(), 'outputs')

    @classmethod
    def get_latest_checkpoint(cls):
        """Retorna o caminho para o checkpoint mais recente gerado pelo ModelCheckpoint."""
        from pathlib import Path
        checkpoints_dir = Path(cls.BASE_PATH) / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Diretório de checkpoints não encontrado: {checkpoints_dir}")
        
        # Pega todos os .ckpt, ordenados por data de modificação
        checkpoints = sorted(checkpoints_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            raise FileNotFoundError(f"Nenhum arquivo .ckpt encontrado em {checkpoints_dir}")
        
        return str(checkpoints[-1])