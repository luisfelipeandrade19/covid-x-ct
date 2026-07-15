"""Script para verificar visualmente a segmentação pulmonar.

Carrega algumas imagens aleatórias de cada classe e mostra lado a lado:
Original | Máscara | Segmentada

Uso: python test_segmentation.py
"""

import os
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend com janela (para visualização local)
import matplotlib.pyplot as plt

from dataset import segment_lungs
from config import Config


def test_segmentation(num_samples=3):
    """Mostra exemplos de segmentação para cada classe."""

    # Lê o arquivo de anotação de teste
    import pandas as pd
    txt_path = os.path.join(Config.BASE_PATH, "test_COVIDx_CT-3A.txt")
    data = pd.read_csv(txt_path, sep=" ", header=None)

    class_names = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}

    # Separa amostras por classe
    samples_by_class = {}
    for cls_id in range(3):
        class_data = data[data[1] == cls_id]
        selected = class_data.sample(n=min(num_samples, len(class_data)), random_state=42)
        samples_by_class[cls_id] = selected

    # Cria o grid: linhas = classes × amostras, colunas = original | máscara | segmentada
    total_rows = num_samples * 3
    fig, axes = plt.subplots(total_rows, 3, figsize=(12, 4 * total_rows))

    row = 0
    for cls_id in range(3):
        for _, sample in samples_by_class[cls_id].iterrows():
            img_name = sample[0]
            img_path = os.path.join(Config.IMAGES_DIR, img_name)

            # Lê em escala de cinza
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Imagem não encontrada: {img_path}")
                continue

            # Aplica segmentação
            segmented = segment_lungs(image)

            # Cria a máscara para visualização (diferença entre original e segmentada)
            mask = (segmented > 0).astype(np.uint8) * 255

            # Plota original
            axes[row, 0].imshow(image, cmap='gray')
            axes[row, 0].set_title(f"{class_names[cls_id]} — Original", fontsize=10)
            axes[row, 0].axis("off")

            # Plota máscara
            axes[row, 1].imshow(mask, cmap='gray')
            axes[row, 1].set_title("Máscara", fontsize=10)
            axes[row, 1].axis("off")

            # Plota segmentada
            axes[row, 2].imshow(segmented, cmap='gray')
            axes[row, 2].set_title("Segmentada", fontsize=10)
            axes[row, 2].axis("off")

            row += 1

    plt.suptitle("Verificação da Segmentação Pulmonar", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Salva e mostra
    output_path = os.path.join(Config.IMG_OUTPUTS_PATH, "test_segmentation.png")
    os.makedirs(Config.IMG_OUTPUTS_PATH, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Imagem salva em: {output_path}")
    plt.show()


if __name__ == "__main__":
    test_segmentation(num_samples=3)
