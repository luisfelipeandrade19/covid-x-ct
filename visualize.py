import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from dataset import val_transformacoes
from loaders import val_loader
from model import SimpleClassifier

CLASSES = ["Normal", "Pneumonia", "COVID-19"]


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def generate_gradcam(model, image_tensor, target_layer, target_class=None):
    """Gera o heatmap Grad-CAM para uma imagem.

    Args:
        model: modelo carregado (SimpleClassifier).
        image_tensor: tensor da imagem (1, C, H, W), já normalizado.
        target_layer: camada convolucional alvo.
        target_class: classe para a qual gerar o mapa. Se None, usa a predita.

    Returns:
        heatmap: numpy array (H, W) normalizado em [0, 1].
        pred_class: índice da classe predita.
    """
    ativacoes = []
    gradientes = []

    def hook_ativacoes(module, input, output):
        ativacoes.append(output.detach())

    def hook_gradientes(module, grad_input, grad_output):
        gradientes.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(hook_ativacoes)
    handle_bwd = target_layer.register_full_backward_hook(hook_gradientes)

    # Forward
    model.eval()
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward na classe alvo
    classe = target_class if target_class is not None else pred_class
    model.zero_grad()
    score = output[0, classe]
    score.backward()

    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # Calcula o heatmap
    grads = gradientes[0]         
    acts = ativacoes[0]          

    # Pesos: global average pooling dos gradientes
    pesos = grads.mean(dim=[2, 3], keepdim=True) 

    # Combinação ponderada
    cam = (pesos * acts).sum(dim=1, keepdim=True)  
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normaliza para [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, pred_class


# ---------------------------------------------------------------------------
# Sobreposição do heatmap na imagem original
# ---------------------------------------------------------------------------
def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """Sobrepõe o heatmap Grad-CAM na imagem original.

    Args:
        original_image: numpy array (H, W, 3) em BGR, valores 0-255.
        heatmap: numpy array (h, w) normalizado em [0, 1].
        alpha: intensidade da sobreposição.

    Returns:
        resultado: numpy array (H, W, 3) em RGB, valores 0-255.
    """
    h, w = original_image.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)


    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


    resultado = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    resultado = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)

    return resultado


# ---------------------------------------------------------------------------
# Desnormaliza imagem para visualização
# ---------------------------------------------------------------------------
def denormalize(tensor):
    """Reverte a normalização ImageNet para visualização."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu().clone() * std + mean
    img = img.clamp(0, 1)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


# ---------------------------------------------------------------------------
# Gera grid de visualizações Grad-CAM por classe
# ---------------------------------------------------------------------------
def generate_gradcam_grid(model, dataloader, target_layer, num_per_class=3):
    """Gera um grid com Grad-CAM para exemplos de cada classe.

    Args:
        model: modelo carregado.
        dataloader: dataloader (val ou test).
        target_layer: camada alvo para Grad-CAM.
        num_per_class: quantas imagens por classe exibir.
    """
    device = next(model.parameters()).device
    num_classes = len(CLASSES)

    exemplos = {c: [] for c in range(num_classes)}
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            c = label.item()
            if len(exemplos[c]) < num_per_class:
                exemplos[c].append(img)
        if all(len(v) >= num_per_class for v in exemplos.values()):
            break

    # Gera o grid: linhas = classes, colunas = (original | gradcam) * num_per_class
    fig, axes = plt.subplots(
        num_classes, num_per_class * 2,
        figsize=(4 * num_per_class * 2, 4 * num_classes),
    )

    for row, classe in enumerate(range(num_classes)):
        for col, img_tensor in enumerate(exemplos[classe]):
            img_input = img_tensor.unsqueeze(0).to(device)

            # Gera Grad-CAM
            heatmap, pred = generate_gradcam(model, img_input, target_layer, target_class=classe)

            img_original = denormalize(img_tensor)
            img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            # Sobreposição
            img_overlay = overlay_heatmap(img_original, heatmap)

            ax_orig = axes[row, col * 2]
            ax_orig.imshow(img_original_rgb)
            ax_orig.set_title(f"{CLASSES[classe]}", fontsize=10)
            ax_orig.axis("off")

            ax_cam = axes[row, col * 2 + 1]
            ax_cam.imshow(img_overlay)
            ax_cam.set_title(f"Pred: {CLASSES[pred]}", fontsize=10)
            ax_cam.axis("off")

    plt.suptitle("Grad-CAM — DenseNet161", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("gradcam_grid.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Salvo em gradcam_grid.png")


if __name__ == "__main__":
    checkpoint_path = os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    model = SimpleClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Camada alvo: último bloco denso (antes do pooling/classifier)
    target_layer = model.model.features.denseblock4

    # Gera grid com 3 exemplos por classe
    generate_gradcam_grid(model, val_loader, target_layer, num_per_class=3)