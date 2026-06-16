import os

import cv2
import matplotlib
matplotlib.use('Agg')  # Backend sem display (compatível com Docker)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from loaders import val_loader
from model import SimpleClassifier

# Nomes das classes para exibição nos gráficos
CLASSES = ["Normal", "Pneumonia", "COVID-19"]


# ---------------------------------------------------------------------------
# Grad-CAM — Geração do heatmap
# ---------------------------------------------------------------------------
def generate_gradcam(model, image_tensor, target_layer, target_class=None):
    """Gera o heatmap Grad-CAM para uma única imagem.

    O Grad-CAM utiliza os gradientes da classe alvo fluindo para a última
    camada convolucional para produzir um mapa de calor que destaca as
    regiões mais importantes da imagem para a decisão do modelo.

    Etapas:
        1. Registra hooks de forward (ativações) e backward (gradientes)
        2. Executa o forward pass da imagem
        3. Executa o backward pass do score da classe alvo
        4. Calcula a média global dos gradientes (pesos por canal)
        5. Combina os pesos com as ativações e aplica ReLU

    Args:
        model: modelo carregado (SimpleClassifier).
        image_tensor: tensor da imagem (1, C, H, W), já normalizado.
        target_layer: camada convolucional alvo (ex: denseblock4).
        target_class: classe para a qual gerar o mapa. Se None, usa a predita.

    Returns:
        heatmap: numpy array (H, W) normalizado em [0, 1].
        pred_class: índice da classe predita.
    """
    # Listas para armazenar ativações e gradientes capturados pelos hooks
    activations = []
    gradients = []

    # Hook de forward: captura as ativações da camada alvo
    def hook_activations(module, input, output):
        activations.append(output.detach())

    # Hook de backward: captura os gradientes da camada alvo
    def hook_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Registra os hooks na camada alvo
    handle_fwd = target_layer.register_forward_hook(hook_activations)
    handle_bwd = target_layer.register_full_backward_hook(hook_gradients)

    # Forward pass
    model.eval()
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    # Define a classe alvo para o backward (predita ou especificada)
    target = target_class if target_class is not None else pred_class

    # Backward pass: calcula os gradientes em relação ao score da classe alvo
    model.zero_grad()
    score = output[0, target]
    score.backward()

    # Remove os hooks para evitar memory leak
    handle_fwd.remove()
    handle_bwd.remove()

    # Recupera as ativações e gradientes capturados
    grads = gradients[0]
    acts = activations[0]

    # Calcula os pesos: média global dos gradientes sobre as dimensões espaciais (H, W)
    weights = grads.mean(dim=[2, 3], keepdim=True)

    # Combinação ponderada: soma dos mapas de ativação multiplicados pelos pesos
    cam = (weights * acts).sum(dim=1, keepdim=True)

    # ReLU: mantém apenas as regiões com influência positiva
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normaliza o heatmap para o intervalo [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, pred_class

# Sobreposição do heatmap na imagem original
def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """Sobrepõe o heatmap Grad-CAM na imagem original.

    Redimensiona o heatmap para o tamanho da imagem, aplica o colormap
    JET (azul → vermelho) e combina com a imagem original usando
    blending ponderado.

    Args:
        original_image: numpy array (H, W, 3) em BGR, valores 0-255.
        heatmap: numpy array (h, w) normalizado em [0, 1].
        alpha: intensidade da sobreposição (0 = só imagem, 1 = só heatmap).

    Returns:
        result: numpy array (H, W, 3) em RGB, valores 0-255.
    """
    h, w = original_image.shape[:2]

    # Redimensiona o heatmap para o tamanho da imagem original
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Aplica o colormap JET (azul=frio → vermelho=quente)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Combina imagem original com o heatmap colorido
    result = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)

    # Converte de BGR para RGB (matplotlib espera RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


# Desnormalização da imagem para visualização
def denormalize(tensor):
    """Reverte a normalização ImageNet para recuperar a imagem visual.

    A normalização aplicada durante o pré-processamento (mean/std do ImageNet)
    precisa ser revertida para exibir a imagem com cores corretas.

    Args:
        tensor: tensor (C, H, W) normalizado com mean/std do ImageNet.

    Returns:
        img_bgr: numpy array (H, W, 3) em BGR, valores 0-255.
    """
    # Parâmetros de normalização do ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Reverte: imagem = (tensor * std) + mean
    img = tensor.cpu().clone() * std + mean
    img = img.clamp(0, 1)     # Garante valores no intervalo [0, 1]

    # Converte de (C, H, W) tensor para (H, W, C) numpy em BGR
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


# Geração do grid de visualizações Grad-CAM por classe
def generate_gradcam_grid(model, dataloader, target_layer, num_per_class=3):
    """Gera um grid visual com Grad-CAM para exemplos de cada classe.

    O grid é organizado como:
        - Linhas: uma por classe (Normal, Pneumonia, COVID-19)
        - Colunas: pares (imagem original | Grad-CAM) para cada amostra

    Args:
        model: modelo carregado e em modo eval.
        dataloader: dataloader de validação ou teste.
        target_layer: camada alvo para o Grad-CAM (ex: denseblock4).
        num_per_class: número de imagens por classe a exibir.
    """
    device = next(model.parameters()).device
    num_classes = len(CLASSES)

    # Coleta amostras de cada classe
    samples = {c: [] for c in range(num_classes)}
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            c = label.item()
            if len(samples[c]) < num_per_class:
                samples[c].append(img)

        # Para quando tiver amostras suficientes de todas as classes
        if all(len(v) >= num_per_class for v in samples.values()):
            break

    # Cria o grid: linhas = classes, colunas = (original | gradcam) × num_per_class
    fig, axes = plt.subplots(
        num_classes, num_per_class * 2,
        figsize=(4 * num_per_class * 2, 4 * num_classes),
    )

    for row, cls in enumerate(range(num_classes)):
        for col, img_tensor in enumerate(samples[cls]):
            img_input = img_tensor.unsqueeze(0).to(device)

            # Gera o heatmap Grad-CAM para a classe correspondente
            heatmap, pred = generate_gradcam(model, img_input, target_layer, target_class=cls)

            # Recupera a imagem original (sem normalização)
            img_original = denormalize(img_tensor)
            img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            # Sobrepõe o heatmap na imagem original
            img_overlay = overlay_heatmap(img_original, heatmap)

            # Plota a imagem original
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(img_original_rgb)
            ax_orig.set_title(f"{CLASSES[cls]}", fontsize=10)
            ax_orig.axis("off")

            # Plota o Grad-CAM sobreposto
            ax_cam = axes[row, col * 2 + 1]
            ax_cam.imshow(img_overlay)
            ax_cam.set_title(f"Pred: {CLASSES[pred]}", fontsize=10)
            ax_cam.axis("off")

    plt.suptitle("Grad-CAM — DenseNet161", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("gradcam_grid.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Salvo em gradcam_grid.png")


# Execução principal
if __name__ == "__main__":
    # Carrega o melhor checkpoint salvo durante o treino
    checkpoint_path = os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    model = SimpleClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Define o dispositivo (GPU se disponível, senão CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Camada alvo: último bloco denso da DenseNet161 (antes do pooling e classificador)
    target_layer = model.model.features.denseblock4

    # Gera grid com 3 exemplos por classe
    generate_gradcam_grid(model, val_loader, target_layer, num_per_class=3)