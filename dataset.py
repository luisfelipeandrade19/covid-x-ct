import logging
import os

import cv2
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def segment_lungs(image):
    """Segmenta os pulmões de uma imagem CT usando processamento de imagem.

    Pipeline:
        1. Binarização com threshold de Otsu (separa regiões escuras/claras)
        2. Operações morfológicas para limpar ruído
        3. Identifica componentes conectados
        4. Remove componentes que tocam as bordas (fundo da imagem)
        5. Mantém os 2 maiores componentes (pulmão esquerdo e direito)
        6. Aplica a máscara na imagem original

    Args:
        image: numpy array (H, W) em escala de cinza.

    Returns:
        Imagem com apenas a região pulmonar, fundo zerado (preto).
    """
    # 1. Binarização: Otsu encontra automaticamente o melhor threshold
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Operações morfológicas para remover ruído pequeno e fechar buracos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Encontra componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 4. Remove componentes que tocam as bordas (são fundo, não pulmão)
    h, w = image.shape
    border_labels = set()
    for label_id in range(1, num_labels):  # Ignora o fundo (label 0)
        x, y, bw, bh = stats[label_id, cv2.CC_STAT_LEFT], stats[label_id, cv2.CC_STAT_TOP], \
                        stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT]
        # Se o componente toca qualquer borda da imagem
        if x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h:
            border_labels.add(label_id)

    # 5. Filtra: mantém apenas componentes internos, ordenados por área
    valid_components = []
    for label_id in range(1, num_labels):
        if label_id not in border_labels:
            area = stats[label_id, cv2.CC_STAT_AREA]
            valid_components.append((label_id, area))

    # Ordena por área decrescente e mantém no máximo os 2 maiores (pulmões)
    valid_components.sort(key=lambda x: x[1], reverse=True)
    keep_labels = [comp[0] for comp in valid_components[:2]]

    # 6. Cria a máscara final com apenas os pulmões
    mask = np.zeros_like(image, dtype=np.uint8)
    for label_id in keep_labels:
        mask[labels == label_id] = 255

    # Aplica a máscara: mantém pulmões, zera o resto
    result = cv2.bitwise_and(image, mask)

    # Fallback: se a segmentação falhou (nenhum componente válido), retorna a original
    if result.max() == 0:
        return image

    return result



def get_dataset_path():
    """Retorna o caminho do dataset COVIDx CT.

    O caminho é lido da variável de ambiente DATASET_PATH.
    Se não estiver definida, lança um erro com instruções.

    Returns:
        Caminho raiz do dataset.
    """
    env_path = os.environ.get("DATASET_PATH")
    if env_path and os.path.isdir(env_path):
        logger.info(f"Usando dataset local: {env_path}")
        return env_path

    raise EnvironmentError(
        "Variável de ambiente DATASET_PATH não definida ou diretório não encontrado. "
        "Defina-a com o caminho do dataset COVIDx CT-3A. "
        "Exemplo: export DATASET_PATH=/app/data"
    )


# Obtém o caminho do dataset via variável de ambiente
ctxcovid = get_dataset_path()


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

    def __init__(self, txt_path, img_dir, transform=None, use_segmented=False):
        # Valida que o arquivo de anotação existe
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(
                f"Arquivo de anotação não encontrado: {txt_path}"
            )

        # Valida que o diretório de imagens existe
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Diretório de imagens não encontrado: {img_dir}"
            )

        # Lê o arquivo de anotação com separador de espaço, sem cabeçalho
        self.data = pd.read_csv(txt_path, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.use_segmented = use_segmented

        # Inicializa o CLAHE para realce adaptativo de contraste
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Log de distribuição de classes para detecção de desbalanceamento
        class_counts = self.data[1].value_counts().sort_index()
        logger.info(
            f"Dataset carregado: {len(self.data)} amostras | "
            f"Distribuição por classe: {dict(class_counts)}"
        )

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
            raise ValueError(
                f"Imagem não encontrada ou corrompida: {img_path}"
            )

        # Aplica segmentação pulmonar algorítmica (se habilitada)
        if self.use_segmented:
           image = segment_lungs(image)

        # Aplica CLAHE para intensificar bordas e diferenças nos tecidos
        try:
            image = self.clahe.apply(image)
        except cv2.error as e:
            logger.warning(
                f"CLAHE falhou para {img_path}: {e}. "
                f"Usando imagem sem realce de contraste."
            )

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