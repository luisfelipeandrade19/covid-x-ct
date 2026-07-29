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
    
    # O Grad-CAM PRECISA de gradientes para funcionar. Como carregamos o checkpoint,
    # as camadas podem ter vindo congeladas do __init__. Vamos destravá-las:
    model.requires_grad_(True)
    
    # Seleciona a última camada convolucional como alvo para o CAM
    # Na DenseNet161, a última Conv2d real é:
    target_layers = [model.model.features.denseblock4.denselayer24.conv2]
    
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

def run_statistical_tests(results, experiment_name):
    """Executa o Bootstrapping para CIs e salva predições para McNemar, exportando para PNG."""
    from sklearn.utils import resample
    from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc, confusion_matrix
    import pandas as pd
    
    all_labels = results["labels"]
    all_preds = results["normal_preds"]
    all_probs = results["normal_probs"][:, 1]
    
    n_bootstraps = 1000
    rng_seed = 42
    bootstrapped_auc = []
    bootstrapped_sens = []
    bootstrapped_spec = []
    
    print(f"Rodando {n_bootstraps} amostras de Bootstrapping...")
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(all_labels)), replace=True, random_state=rng_seed + i)
        if len(np.unique(all_labels[indices])) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(all_labels[indices], all_probs[indices])
        bootstrapped_auc.append(auc(fpr, tpr))
        
        bootstrapped_sens.append(recall_score(all_labels[indices], all_preds[indices]))
        
        tn, fp, fn, tp = confusion_matrix(all_labels[indices], all_preds[indices]).ravel()
        bootstrapped_spec.append(tn / (tn + fp))

    def get_ci_str(name, scores):
        scores = np.array(scores)
        return f"{name:<15}: {scores.mean():.4f} (95% CI: {np.percentile(scores, 2.5):.4f} - {np.percentile(scores, 97.5):.4f})"

    # Monta o texto de saída
    text_lines = [
        "Resultados do Bootstrapping (1000 amostras)",
        "-" * 55,
        get_ci_str("AUC", bootstrapped_auc),
        get_ci_str("Sensibilidade", bootstrapped_sens),
        get_ci_str("Especificidade", bootstrapped_spec),
        "-" * 55
    ]
    
    final_text = "\n".join(text_lines)
    print(final_text)
    
    # Salva o texto como PNG para o WandB
    save_text_as_png(final_text, "bootstrapping_metrics.png", title=f"Estatística - {experiment_name}")
    
    # Salva as predições em CSV para McNemar cruzado no futuro
    df_preds = pd.DataFrame({'real': all_labels, 'pred_atual': all_preds})
    csv_path = os.path.join(Config.IMG_OUTPUTS_PATH, f"preds_{experiment_name}.csv")
    df_preds.to_csv(csv_path, index=False)
    print(f"Predições salvas em {csv_path} para futuros testes de McNemar.")
