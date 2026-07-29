import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Aviso: pytorch-grad-cam não instalado. Rode: pip install grad-cam")

def save_text_as_png(text, filename, title="Métricas"):
    """Renderiza um texto (como o classification_report) em um PNG de alta resolução."""
    out_dir = Config.IMG_OUTPUTS_PATH
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Adiciona o título
    ax.text(0.5, 0.95, title, fontsize=16, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Renderiza o texto mono-espaçado
    ax.text(0.05, 0.85, text, fontsize=12, family='monospace', va='top', ha='left', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Texto salvo como PNG: {out_path}")


def generate_cams(model, dataloader, device, num_images=4):
    """Gera mapas de ativação (Grad-CAM, Grad-CAM++, HiResCAM) para um lote de imagens."""
    out_dir = Config.IMG_OUTPUTS_PATH
    os.makedirs(out_dir, exist_ok=True)
    
    model.eval()
    
    # Seleciona a última camada convolucional como alvo para o CAM
    # Na DenseNet161, a última camada de features antes do Pooling/CBAM é:
    # model.model.features.norm5 ou model.model.features.denseblock4.denselayer24.conv2
    target_layers = [model.model.features.norm5]
    
    cam_methods = {
        "Grad-CAM": GradCAM,
        "Grad-CAM++": GradCAMPlusPlus,
        "HiResCAM": HiResCAM
    }
    
    # Extrai o primeiro lote do DataLoader
    images, labels = next(iter(dataloader))
    
    # Limita o número de imagens
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)
    
    for cam_name, cam_class in cam_methods.items():
        try:
            # Inicializa o CAM
            cam = cam_class(model=model, target_layers=target_layers)
            
            # Alvos (queremos explicar a predição da classe COVID-19 = 1)
            targets = [ClassifierOutputTarget(1) for _ in range(images.size(0))]
            
            # Gera a máscara (batch_size, H, W)
            grayscale_cams = cam(input_tensor=images, targets=targets)
            
            # Prepara a figura
            fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
            if num_images == 1:
                axes = [axes]
                
            for i in range(num_images):
                # Imagem original para visualização
                # Desnormaliza: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Sobrepõe o heatmap na imagem original
                visualization = show_cam_on_image(img, grayscale_cams[i], use_rgb=True)
                
                real_lbl = "COVID-19" if labels[i].item() == 1 else "Pneumonia"
                
                axes[i].imshow(visualization)
                axes[i].set_title(f"Real: {real_lbl}", fontsize=14, fontweight='bold')
                axes[i].axis('off')
                
            plt.suptitle(f"Mapas de Ativação: {cam_name}", fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            out_path = os.path.join(out_dir, f"{cam_name.lower().replace('+', 'p')}_samples.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"{cam_name} salvo em: {out_path}")
            
        except Exception as e:
            print(f"Falha ao gerar {cam_name}: {e}")
