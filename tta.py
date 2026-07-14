import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Backend sem display (compatível com Docker)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import seaborn as sns

from config import Config
from loaders import test_loader
from model import SimpleClassifier

logger = logging.getLogger(__name__)



def get_tta_transforms():
    """Retorna a lista de transformações para Test-Time Augmentation.

    Cada transformação é uma função que recebe um tensor (C, H, W)
    já normalizado e retorna o tensor transformado.

    A primeira transformação é sempre a identidade (imagem original),
    garantindo que a predição base sempre faça parte da agregação.

    Returns:
        Lista de funções de transformação.
    """
    return [
        lambda x: x,                                        # Original (identidade)
        lambda x: TF.hflip(x),                              # Flip horizontal
        lambda x: TF.rotate(x, angle=5),                    # Rotação +5°
        lambda x: TF.rotate(x, angle=-5),                   # Rotação -5°
        lambda x: TF.rotate(x, angle=10),                   # Rotação +10°
        lambda x: TF.rotate(x, angle=-10),                  # Rotação -10°
        lambda x: TF.adjust_brightness(x, brightness_factor=1.1),  # Brilho +10%
        lambda x: TF.adjust_brightness(x, brightness_factor=0.9),  # Brilho -10%
    ]

def predict_with_tta(model, image_tensor, device, tta_transforms):
    """Realiza inferência com Test-Time Augmentation em uma única imagem.

    Aplica cada transformação na imagem, coleta as probabilidades
    softmax de cada versão, e retorna a média das probabilidades.

    Args:
        model: modelo carregado em modo eval.
        image_tensor: tensor da imagem (C, H, W), já normalizado.
        device: dispositivo (CPU ou GPU).
        tta_transforms: lista de funções de transformação.

    Returns:
        avg_probs: tensor (C,) com probabilidades médias por classe.
    """
    all_probs = []

    for transform in tta_transforms:
        # Aplica a transformação e adiciona a dimensão de batch
        augmented = transform(image_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(augmented)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)

    # Empilha todas as probabilidades e calcula a média
    stacked = torch.cat(all_probs, dim=0)       # (N_transforms, C)
    avg_probs = stacked.mean(dim=0)              # (C,)

    return avg_probs

def evaluate_with_tta(model, dataloader, device):
    """Avalia o modelo no conjunto de teste com e sem TTA.

    Coleta predições normais e com TTA lado a lado para
    permitir comparação direta de métricas.

    Args:
        model: modelo carregado em modo eval.
        dataloader: DataLoader do conjunto de teste.
        device: dispositivo (CPU ou GPU).

    Returns:
        Dicionário com predições e rótulos para ambos os modos.
    """
    tta_transforms = get_tta_transforms()
    class_names = ["Normal", "Pneumonia", "COVID-19"]

    # Listas para acumular resultados
    normal_preds, normal_probs = [], []
    tta_preds, tta_probs = [], []
    all_labels = []

    logger.info(f"Avaliando com TTA ({len(tta_transforms)} transformações por imagem)...")

    for batch_idx, (images, labels) in enumerate(dataloader):
        for img, label in zip(images, labels):
            # Predição NORMAL (sem TTA)
            with torch.no_grad():
                logits = model(img.unsqueeze(0).to(device))
                probs_normal = torch.softmax(logits, dim=1).squeeze(0)

            normal_probs.append(probs_normal.cpu().numpy())
            normal_preds.append(probs_normal.argmax().item())

            # Predição com TTA (média de N augmentações)
            probs_tta = predict_with_tta(model, img, device, tta_transforms)
            tta_probs.append(probs_tta.cpu().numpy())
            tta_preds.append(probs_tta.argmax().item())

            all_labels.append(label.item())

    return {
        "labels": np.array(all_labels),
        "normal_preds": np.array(normal_preds),
        "normal_probs": np.array(normal_probs),
        "tta_preds": np.array(tta_preds),
        "tta_probs": np.array(tta_probs),
    }

def main():
    """Executa a avaliação com Test-Time Augmentation.

    Gera: tta_comparison.png.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega o melhor checkpoint salvo durante o treino
    checkpoint_path = os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    model = SimpleClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    class_names = ["Normal", "Pneumonia", "COVID-19"]

    # Avalia com e sem TTA
    results = evaluate_with_tta(model, test_loader, device)

    # Relatório de classificação — Sem TTA
    logger.info("=== Resultados SEM TTA ===")
    print(classification_report(
        results["labels"], results["normal_preds"],
        target_names=class_names, digits=4,
    ))

    # Relatório de classificação — Com TTA
    logger.info("=== Resultados COM TTA ===")
    print(classification_report(
        results["labels"], results["tta_preds"],
        target_names=class_names, digits=4,
    ))

    # Matrizes de confusão lado a lado
    caminho_outputs = Path(Config.IMG_OUTPUTS_PATH)
    caminho_outputs.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Matriz de confusão — Sem TTA
    cm_normal = confusion_matrix(results["labels"], results["normal_preds"])
    sns.heatmap(cm_normal, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Sem TTA")
    ax1.set_xlabel("Predição")
    ax1.set_ylabel("Real")

    # Matriz de confusão — Com TTA
    cm_tta = confusion_matrix(results["labels"], results["tta_preds"])
    sns.heatmap(cm_tta, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title("Com TTA (8 augmentações)")
    ax2.set_xlabel("Predição")
    ax2.set_ylabel("Real")

    plt.suptitle("Comparação de Matrizes de Confusão — Normal vs TTA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(caminho_outputs / "tta_comparison.png", dpi=300)
    plt.close()

    logger.info("TTA concluído.")


if __name__ == "__main__":
    main()
