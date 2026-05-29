import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from loaders import val_loader
from model import SimpleClassifier

if __name__ == "__main__":
    # Carrega o modelo do checkpoint salvo pelo treinamento
    checkpoint_path = os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    model = SimpleClassifier.load_from_checkpoint(checkpoint_path)

    print("\n--- AVALIAÇÃO FINAL ---")
    model.eval()
    todas_preds, todas_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            todas_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            todas_labels.extend(y.numpy())

    print(
        classification_report(
            todas_labels,
            todas_preds,
            target_names=["Normal", "Pneumonia", "COVID-19"],
            digits=4,
        )
    )

    # Matriz confusão
    cm = confusion_matrix(todas_labels, todas_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia", "COVID-19"],
        yticklabels=["Normal", "Pneumonia", "COVID-19"],
    )
    plt.title("Matriz de Confusão")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("matriz-confusao.png")
    plt.show()

    # Matriz confusão normalizada
    cm_norm = confusion_matrix(todas_labels, todas_preds, normalized="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia", "COVID-19"],
        yticklabels=["Normal", "Pneumonia", "COVID-19"],
    )
    plt.title("Matriz de Confusão Normalizada")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("matriz-confusao-normalizada.png")
    plt.show()

    # Curvas de Treinamento

    # ALERTA: Sempre trocar a versão quando treinar novamente
    metrics_path = (
        Path(Config.BASE_PATH) / "lightning_csv_logs" / "version_0" / "metrics.csv"
    )
    metrics = pd.read_csv(metrics_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(
        metrics["epoch"].dropna().unique(),
        metrics.grouphy("epoch")["train_loss"].mean().dropna(),
        label="Treino",
    )
    ax1.plot(
        metrics["epoch"].dropna().unique(),
        metrics.grouphy("epoch")["val_loss"].mean().dropna(),
        label="Validação",
    )
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.set_title("Evolução da Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Acuracy
    ax2.plot(
        metrics["epoch"].dropna().unique(),
        metrics.grouphy("epoch")["train_acc"].mean().dropna(),
        label="Treino",
    )
    ax1.plot(
        metrics["epoch"].dropna().unique(),
        metrics.grouphy("epoch")["val_acc"].mean().dropna(),
        label="Validação",
    )
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.set_title("Evolução da Acuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.save_fig("curvas_treinamento.png", dpi=300)
