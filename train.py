import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from callbacks import GradualUnfreezing

from config import Config
from loaders import train_loader, val_loader
from model import SimpleClassifier

# Configura o logging para todo o projeto
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Seed — garante reprodutibilidade
    
    pl.seed_everything(Config.SEED, workers=True)

    # Callbacks — controlam comportamento durante o treino

    # Barra de progresso rica (visual aprimorado no terminal)
    rich_progress = RichProgressBar()

    # Early stopping: para o treino se val_loss não melhorar por 5 épocas
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Descongelamento gradual da backbone a cada N épocas
    gradual_unfreeze = GradualUnfreezing(
        epochs_per_stage=Config.EPOCHS_PER_STAGE,
        max_stage=Config.MAX_UNFREEZE_STAGE,
    )

    # Salva o melhor modelo com base na val_loss
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(Config.BASE_PATH, "checkpoints"),
        filename="best_model",
        save_top_k=1,           # Mantém apenas o melhor checkpoint
        mode="min",
    )

    # Modelo e Logger

    # Instancia o classificador com os hiperparâmetros definidos no Config
    model = SimpleClassifier(
        num_classes=Config.NUM_CLASSES, learning_rate=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    )

    # Logger CSV: salva métricas de treino/validação a cada época
    csv_logger = CSVLogger(
        save_dir=Config.BASE_PATH,
        name="lightning_csv_logs",
    )

    # Trainer — orquestra o loop de treino
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,        # Número máximo de épocas
        accelerator="gpu",                   # Utiliza GPU para aceleração
        devices=1,                           # Número de GPUs a usar
        callbacks=[rich_progress, early_stop, model_checkpoint, gradual_unfreeze],
        logger=csv_logger,
        precision="16-mixed",                # Precisão mista para economia de VRAM              
    )

    # Inicia o treinamento com os dataloaders de treino e validação
    logger.info("Iniciando treinamento...")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Treinamento concluído.")
