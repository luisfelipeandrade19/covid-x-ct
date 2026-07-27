from config import Config
from dataset import CovidCTDataset, train_transforms, val_transforms
from torch.utils.data import DataLoader

# Datasets — instâncias do CovidCTDataset para cada split

# Dataset de treino com augmentation (flip, rotação, jitter)
train_dataset = CovidCTDataset(
    Config.TRAIN_TXT,
    Config.IMAGES_DIR_TRAIN,
    transform=train_transforms,
)

# Dataset de validação sem augmentation
val_dataset = CovidCTDataset(
    Config.VAL_TXT,
    Config.IMAGES_DIR_TRAIN,       # Val usa o mesmo diretório de imagens que treino
    transform=val_transforms,
)

# Dataset de teste sem augmentation
test_dataset = CovidCTDataset(
    Config.TEST_TXT,
    Config.IMAGES_DIR_TEST,        # Teste pode usar diretório diferente (segmentação)
    transform=val_transforms,
)

# DataLoaders — iteradores de lote para o treinamento

# Loader de treino com shuffle (embaralhamento a cada época)
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,          # Acelera transferência CPU → GPU
)

# Loader de validação sem shuffle (ordem fixa para reprodutibilidade)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)

# Loader de teste sem shuffle
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)
