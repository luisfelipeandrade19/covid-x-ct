from config import Config
from dataset import CovidCTDataset, train_transforms, val_transforms
from torch.utils.data import DataLoader
import os

# ---------------------------------------------------------------------------
# Datasets — instâncias do CovidCTDataset para cada split
# ---------------------------------------------------------------------------

# Dataset de treino com augmentation (flip, rotação, jitter)
train_dataset = CovidCTDataset(
    os.path.join(Config.BASE_PATH, "train_COVIDx_CT-3A.txt"),
    Config.IMAGES_DIR,
    transform=train_transforms,
)

# Dataset de validação sem augmentation
val_dataset = CovidCTDataset(
    os.path.join(Config.BASE_PATH, "val_COVIDx_CT-3A.txt"),
    Config.IMAGES_DIR,
    transform=val_transforms,
)

# Dataset de teste sem augmentation
test_dataset = CovidCTDataset(
    os.path.join(Config.BASE_PATH, "test_COVIDx_CT-3A.txt"),
    Config.IMAGES_DIR,
    transform=val_transforms,
)

# ---------------------------------------------------------------------------
# DataLoaders — iteradores de lote para o treinamento
# ---------------------------------------------------------------------------

# Loader de treino com shuffle (embaralhamento a cada época)
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,          # Acelera transferência CPU → GPU
)

# Loader de validação sem shuffle (ordem fixa para reprodutibilidade)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Loader de teste sem shuffle
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
