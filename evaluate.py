import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Backend sem display (compatível com Docker)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

from config import Config
from loaders import test_loader
from model import SimpleClassifier

if __name__ == "__main__":
    caminho_outputs = Path(Config.IMG_OUTPUTS_PATH)
    caminho_outputs.mkdir(parents=True, exist_ok=True)

    # Carrega o melhor checkpoint salvo durante o treino
    checkpoint_path = os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    model = SimpleClassifier.load_from_checkpoint(checkpoint_path)


    print("\n--- AVALIAÇÃO FINAL ---")
    model.eval()    # Coloca o modelo em modo de avaliação.

    # Listas para acumular predições e rótulos reais
    all_preds, all_probs, all_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inferência 
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            # Calcula probabilidades e predições absolutas
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    # Converte probabilidades para array numpy para facilitar indexação nas curvas
    all_probs = np.array(all_probs)

    # Relatório de classificação (Precision, Recall, F1-Score)
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["Normal", "Pneumonia", "COVID-19"],
            digits=4,
        )
    )

    # Matriz de confusão (valores absolutos)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",                    # Formato inteiro
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia", "COVID-19"],
        yticklabels=["Normal", "Pneumonia", "COVID-19"],
    )
    plt.title("Matriz de Confusão")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(caminho_outputs / "confusion_matrix.png", dpi=300)
    plt.close()

    # Matriz de confusão normalizada (proporções por classe real)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",                  # Formato float com 2 casas decimais
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia", "COVID-19"],
        yticklabels=["Normal", "Pneumonia", "COVID-19"],
    )
    plt.title("Matriz de Confusão Normalizada")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(caminho_outputs / "confusion_matrix_normalized.png", dpi=300)
    plt.close()

    # Curvas de treinamento (Loss e Accuracy por época)

    # Detecta automaticamente a última versão do CSVLogger
    logs_dir = Path(Config.BASE_PATH) / "lightning_csv_logs"
    versions = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("version_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not versions:
        raise FileNotFoundError(f"Nenhuma versão encontrada em {logs_dir}")

    # Lê as métricas da última versão
    metrics_path = versions[-1] / "metrics.csv"
    metrics = pd.read_csv(metrics_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    metrics_by_epoch = metrics.groupby("epoch").mean()

    train_loss = metrics_by_epoch["train_loss"].dropna()
    val_loss = metrics_by_epoch["val_loss"].dropna()
    train_acc = metrics_by_epoch["train_acc"].dropna()
    val_acc = metrics_by_epoch["val_acc"].dropna()

    # Gráfico de Loss (treino vs validação)

    if not train_loss.empty:
        ax1.plot(train_loss.index, train_loss.values, label="Treino")
    if not val_loss.empty:
        ax1.plot(val_loss.index, val_loss.values, label="Validação")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Evolução da Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico de Accuracy (treino vs validação)
    if not train_acc.empty:
        ax2.plot(train_acc.index, train_acc.values, label="Treino")
    if not val_acc.empty:
        ax2.plot(val_acc.index, val_acc.values, label="Validação")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Evolução da Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho_outputs / "training_curves.png", dpi=300)

   
    # --- Curvas ROC e Precision-Recall (Teste) ---

    # Binariza os rótulos para cálculo per-classe
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
    class_names = ["Normal", "Pneumonia", "COVID-19"]

    # Plota a curva ROC para cada classe
    plt.figure(figsize=(8, 6))
    for i in range(Config.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

    # Linha diagonal de referência (classificador aleatório)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC — One vs Rest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_outputs / 'roc_curve.png', dpi=300)

    # Curva Precision-Recall (uma curva por classe)
    plt.figure(figsize=(8, 6))
    for i in range(Config.NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
        ap = average_precision_score(labels_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_outputs / 'precision_recall_curve.png', dpi=300)