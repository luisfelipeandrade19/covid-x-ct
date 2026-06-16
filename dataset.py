import os

import cv2
import kagglehub
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

if __name__ == "__main__":
    # Download do dataset via KaggleHub
    ctxcovid = kagglehub.dataset_download("hgunraj/covidxct")
    print("Data source import complete.")


    class CovidCTDataset(Dataset):
        """Dataset personalizado para imagens de Tomografia Computadorizada (CT).

        Carrega imagens a partir de um arquivo de anotação (.txt), aplica
        pré-processamento com CLAHE para realce de contraste e retorna
        pares (imagem, rótulo).

        Args:
            txt_path: caminho para o arquivo de anotação (espaço como separador).
            img_dir: diretório raiz onde estão as imagens.
            transform: transformações do torchvision a serem aplicadas.
        """

        def __init__(self, txt_path, img_dir, transform=None):
            # Lê o arquivo de anotação com separador de espaço, sem cabeçalho
            self.data = pd.read_csv(txt_path, sep=" ", header=None)
            self.img_dir = img_dir
            self.transform = transform

            # Inicializa o CLAHE para realce adaptativo de contraste
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def __len__(self):
            """Retorna o número total de amostras no dataset."""
            return len(self.data)

        def __getitem__(self, idx):
            """Retorna a imagem pré-processada e seu rótulo pelo índice.

            Etapas do pipeline:
                1. Leitura em escala de cinza (IMREAD_GRAYSCALE)
                2. Aplicação do CLAHE para realce de bordas e tecidos
                3. Conversão para RGB (3 canais) para compatibilidade com DenseNet
                4. Aplicação das transformações (resize, augmentation, normalização)

            Args:
                idx: índice da amostra no dataset.

            Returns:
                Tupla (imagem_transformada, rótulo).
            """
            # Extrai o nome do arquivo e o rótulo da linha correspondente
            img_name = self.data.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)
            label = int(self.data.iloc[idx, 1])

            # Lê a imagem em escala de cinza
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Image not found: {img_path}")

            # Aplica CLAHE para intensificar bordas e diferenças nos tecidos
            image = self.clahe.apply(image)

            # Converte de grayscale para RGB (DenseNet espera 3 canais)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_pil = Image.fromarray(image)

            # Aplica as transformações (resize, augmentation, normalização)
            if self.transform:
                image_pil = self.transform(image_pil)

            return image_pil, label


    # Transformações de treino com data augmentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),           # Redimensiona para o tamanho esperado pela DenseNet
            transforms.RandomHorizontalFlip(0.5),    # Flip horizontal aleatório (50% de chance)
            transforms.RandomRotation(degrees=10),   # Rotação aleatória de até ±10 graus
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variação de brilho e contraste
            transforms.ToTensor(),                   # Converte para tensor PyTorch [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalização ImageNet
        ]
    )

    # Transformações de validação/teste sem augmentation
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),           # Redimensiona para o tamanho esperado
            transforms.ToTensor(),                   # Converte para tensor PyTorch [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalização ImageNet
        ]
    )