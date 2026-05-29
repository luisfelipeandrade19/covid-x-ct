import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from callbacks import GradualUnfreezing

from config import Config
from loaders import train_loader, val_loader
from model import SimpleClassifier


if __name__ == "__main__":
    # Callbacks
    rich_progress = RichProgressBar()
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    gradual_unfreeze = GradualUnfreezing(
        epochs_per_stage=Config.EPOCHS_PER_STAGE,
        max_stage=Config.MAX_UNFREEZE_STAGE,
    )
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(Config.BASE_PATH, "checkpoints"),
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    model = SimpleClassifier(
        num_classes=Config.NUM_CLASSES, learning_rate=Config.LEARNING_RATE
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[rich_progress, early_stop, model_checkpoint, gradual_unfreeze],
        precision="16-mixed",
    )

    trainer.fit(model, train_loader, val_loader)
    model = SimpleClassifier.load_from_checkpoint(
        os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    )
