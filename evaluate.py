import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from loaders import val_loader
from train import model

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

# Heatmap
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
plt.savefig("matriz-confusao.png")
plt.show()
