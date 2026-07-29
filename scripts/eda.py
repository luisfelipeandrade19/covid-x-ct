import os
import sys
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config

def count_classes(base_path, folder_name):
    """Conta a quantidade de imagens por classe em uma determinada pasta."""
    pneumonia_path = os.path.join(base_path, folder_name, 'Pneumonia')
    covid_path = os.path.join(base_path, folder_name, 'COVID-19')
    
    # Se a pasta não existir, retorna 0
    pneu_count = len(glob.glob(os.path.join(pneumonia_path, '*.*'))) if os.path.exists(pneumonia_path) else 0
    covid_count = len(glob.glob(os.path.join(covid_path, '*.*'))) if os.path.exists(covid_path) else 0
    
    return pneu_count, covid_count

def plot_class_distribution(out_dir):
    """Gera o gráfico de distribuição de classes (Treino vs Teste)."""
    pneu_train, covid_train = count_classes(Config.BASE_PATH, '3A_images')
    pneu_test, covid_test = count_classes(Config.BASE_PATH, '3A_test_images')
    
    labels = ['Treino (Pneumonia)', 'Treino (COVID-19)', 'Teste (Pneumonia)', 'Teste (COVID-19)']
    counts = [pneu_train, covid_train, pneu_test, covid_test]
    colors = ['#1f77b4', '#d62728', '#aec7e8', '#ff9896']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=colors)
    
    # Adiciona os números no topo das barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom', fontweight='bold')
        
    plt.title("Distribuição de Classes no Dataset COVIDx CT-2A", fontsize=16, fontweight='bold')
    plt.ylabel("Número de Imagens")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_class_distribution.png"), dpi=300)
    plt.close()
    print(f"Total Treino: {pneu_train + covid_train} | Total Teste: {pneu_test + covid_test}")

def plot_intensity_histograms(out_dir):
    """Compara o histograma de pixels entre uma imagem FULL e sua versão SEGMENTED."""
    # Pega uma imagem aleatória de treino (Pneumonia)
    full_dir = os.path.join(Config.BASE_PATH, '3A_images', 'Pneumonia')
    seg_dir = os.path.join(Config.BASE_PATH, '3A_images_segmented', 'Pneumonia')
    
    if not os.path.exists(full_dir) or not os.path.exists(seg_dir):
        print("Pastas de imagem não encontradas para gerar histograma.")
        return
        
    full_images = glob.glob(os.path.join(full_dir, '*.*'))
    if not full_images:
        return
        
    sample_full_path = full_images[0]
    filename = os.path.basename(sample_full_path)
    sample_seg_path = os.path.join(seg_dir, filename)
    
    if not os.path.exists(sample_seg_path):
        sample_seg_path = glob.glob(os.path.join(seg_dir, '*.*'))[0] # Pega qualquer uma se o nome não bater
        
    img_full = cv2.imread(sample_full_path, cv2.IMREAD_GRAYSCALE)
    img_seg = cv2.imread(sample_seg_path, cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img_full, cmap='gray')
    axes[0, 0].set_title("Original (Full CT)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_seg, cmap='gray')
    axes[0, 1].set_title("Segmentada (Pulmão Isolado)")
    axes[0, 1].axis('off')
    
    # Histogramas (ignorando o pixel 0 puramente preto para ver melhor a distribuição de tecidos)
    axes[1, 0].hist(img_full[img_full > 0].ravel(), bins=50, color='gray', alpha=0.7)
    axes[1, 0].set_title("Histograma de Intensidade (Full)")
    axes[1, 0].set_xlabel("Valor do Pixel")
    axes[1, 0].set_ylabel("Frequência")
    
    axes[1, 1].hist(img_seg[img_seg > 0].ravel(), bins=50, color='blue', alpha=0.7)
    axes[1, 1].set_title("Histograma de Intensidade (Segmentada)")
    axes[1, 1].set_xlabel("Valor do Pixel")
    
    plt.suptitle("Impacto da Segmentação na Distribuição de Pixels (Ruído vs Parênquima)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_pixel_histogram.png"), dpi=300)
    plt.close()

def plot_sample_collage(out_dir):
    """Cria uma colagem visual das patologias para colocar no paper."""
    categories = [
        ('3A_images/Pneumonia', 'Pneumonia (Original)'),
        ('3A_images/COVID-19', 'COVID-19 (Original)'),
        ('3A_images_segmented/Pneumonia', 'Pneumonia (Segmentada)'),
        ('3A_images_segmented/COVID-19', 'COVID-19 (Segmentada)')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Amostras do Dataset COVIDx CT-2A", fontsize=18, fontweight='bold')
    
    for row in range(2):
        for col in range(4):
            cat_path, cat_title = categories[col]
            full_path = os.path.join(Config.BASE_PATH, cat_path)
            images = glob.glob(os.path.join(full_path, '*.*'))
            
            if images:
                # Pega imagens diferentes para a linha 0 e linha 1
                img_path = images[row] if len(images) > row else images[0]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                axes[row, col].imshow(img, cmap='gray')
            
            if row == 0:
                axes[row, col].set_title(cat_title, fontsize=14)
            axes[row, col].axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_samples_collage.png"), dpi=300)
    plt.close()

def main():
    print("Iniciando Análise Exploratória de Dados (EDA)...")
    out_dir = os.path.join(os.getcwd(), 'outputs', 'EDA')
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Calculando distribuição de classes...")
    plot_class_distribution(out_dir)
    
    print("2. Gerando análise de histograma de pixels (Original vs Segmentado)...")
    plot_intensity_histograms(out_dir)
    
    print("3. Criando grade de amostras visuais...")
    plot_sample_collage(out_dir)
    
    print(f"EDA concluída com sucesso! Verifique a pasta: {out_dir}")

if __name__ == "__main__":
    main()
