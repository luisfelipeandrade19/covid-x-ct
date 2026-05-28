import os

import cv2
import kagglehub
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


# Download do Dataset
ctxcovid = kagglehub.dataset_download("hgunraj/covidxct")
print("Data source import complete.")


class CovidCTDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        self.data = pd.read_csv(txt_path, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.data.iloc[idx, 1])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Imagem não encontrada: {img_path}")

        image = self.clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_pil = Image.fromarray(image)

        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, label


# Transformações com Data Augmentation para Treino
train_transformacoes = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Transformações sem Augmentation para Validação
val_transformacoes = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
