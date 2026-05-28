from config import Config
from dataset import CovidCTDataset, train_transformacoes, val_transformacoes
from torch.utils.data import DataLoader
import os

# Datasets e Loaders
train_dataset = CovidCTDataset(
    os.path.join(Config.BASE_PATH, "train_COVIDx_CT-3A.txt"),
    Config.IMAGES_DIR,
    transform=train_transformacoes,
)
val_dataset = CovidCTDataset(
    os.path.join(Config.BASE_PATH, "val_COVIDx_CT-3A.txt"),
    Config.IMAGES_DIR,
    transform=val_transformacoes,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)