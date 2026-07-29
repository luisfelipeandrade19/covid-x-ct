from src.config import Config
from dataset.dataset import CovidCTDataset, train_transforms, val_transforms
from torch.utils.data import DataLoader
import cv2

# Evita conflitos de threading entre o OpenCV e os workers do DataLoader do PyTorch
cv2.setNumThreads(0)

# Datasets — instâncias do CovidCTDataset para cada split

def get_dataloaders():
    """Retorna (train_loader, val_loader, test_loader) instanciados de acordo com o Config atual."""
    # Recalcula caminhos baseados na flag atual
    images_dir_train = Config.BASE_PATH + ('/3A_images_segmented' if Config.USE_SEGMENTED_DATA else '/3A_images')
    images_dir_test = Config.BASE_PATH + ('/3A_test_images_segmented' if Config.USE_SEGMENTED_DATA else '/3A_images')

    train_dataset = CovidCTDataset(
        Config.TRAIN_TXT,
        images_dir_train,
        transform=train_transforms,
        is_segmented=Config.USE_SEGMENTED_DATA
    )

    val_dataset = CovidCTDataset(
        Config.VAL_TXT,
        images_dir_train,
        transform=val_transforms,
        is_segmented=Config.USE_SEGMENTED_DATA
    )

    test_dataset = CovidCTDataset(
        Config.TEST_TXT,
        images_dir_test,
        transform=val_transforms,
        is_segmented=Config.USE_SEGMENTED_DATA
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

# Mantém instâncias globais para compatibilidade com notebooks isolados,
# inicializando-as com a configuração padrão do config.py
train_loader, val_loader, test_loader = get_dataloaders()
