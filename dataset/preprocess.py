"""Script para pré-processamento offline do dataset.

Este script lê os arquivos .txt (treino, validação e teste), carrega as imagens,
aplica o CLAHE, recorta as bordas (se for a versão segmentada) e salva as imagens
em um novo diretório para que o treinamento via Jupyter seja muito mais rápido.
"""

import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(img_path, is_segmented):
    """Lê a imagem, corta bordas (se segmentada) e aplica CLAHE."""
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    if is_segmented:
        coords = cv2.findNonZero(image)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            image = image[y:y+h, x:x+w]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    try:
        image = clahe.apply(image)
    except Exception as e:
        logger.warning(f"Erro no CLAHE em {img_path}: {e}")

    # Converte para RGB para salvar
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def process_dataset_split(txt_file, input_dir, output_dir, is_segmented):
    """Processa todas as imagens listadas em um arquivo .txt e salva no diretório de saída."""
    if not txt_file.exists():
        logger.error(f"Arquivo não encontrado: {txt_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(txt_file, "r") as f:
        lines = f.readlines()

    logger.info(f"Processando {len(lines)} imagens para {output_dir.name} (Segmentado: {is_segmented})...")
    
    for line in tqdm(lines, desc=f"Salvando em {output_dir.name}"):
        img_name = line.strip().split()[0]
        input_path = input_dir / img_name
        output_path = output_dir / img_name
        
        # Se a imagem já foi processada antes, pula
        if output_path.exists():
            continue

        img_processed = process_image(input_path, is_segmented)
        if img_processed is not None:
            cv2.imwrite(str(output_path), img_processed)
        else:
            logger.warning(f"Imagem falhou ao processar: {input_path}")

def main():
    base = Path(Config.BASE_PATH)
    
    # Textos gerados pelo filtro binário
    txts = [
        base / "train_filtered.txt",
        base / "val_filtered.txt",
        base / "test_filtered.txt"
    ]
    
    # 1. Pré-processar a versão FULL (não segmentada)
    input_full = base / "3A_images"
    output_full = base / "processed_full"
    logger.info("="*50)
    logger.info("PROCESSANDO VERSÃO FULL (NÃO SEGMENTADA)")
    logger.info("="*50)
    for txt in txts:
        process_dataset_split(txt, input_full, output_full, is_segmented=False)

    # 2. Pré-processar a versão SEGMENTADA
    output_seg = base / "processed_segmented"
    logger.info("="*50)
    logger.info("PROCESSANDO VERSÃO SEGMENTADA")
    logger.info("="*50)
    for txt in txts:
        with open(txt, "r") as f:
            for line in tqdm(f.readlines(), desc=f"Segmentadas: {txt.name}"):
                img_name = line.strip().split()[0]
                out_p = output_seg / img_name
                if out_p.exists(): continue
                
                possibles = [
                    base / "3A_images_segmented" / img_name,
                    base / "3A_test_images_segmented" / img_name,
                    base / "3A_test_images_segmented2" / img_name
                ]
                
                img_processed = None
                for p in possibles:
                    if p.exists():
                        img_processed = process_image(p, is_segmented=True)
                        break
                
                if img_processed is not None:
                    output_seg.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_p), img_processed)
                else:
                    logger.warning(f"Imagem segmentada não encontrada: {img_name}")

if __name__ == "__main__":
    main()
