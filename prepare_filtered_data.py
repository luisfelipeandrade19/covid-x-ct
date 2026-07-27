"""Gera arquivos de anotação filtrados para comparação justa.

Cruza as imagens disponíveis nas pastas segmentadas com as anotações originais,
criando arquivos filtrados que contêm apenas imagens com versão segmentada.
Divide o treino em train+val (80/20, estratificado por classe).

Os arquivos gerados são usados por AMBOS os experimentos (original filtrado e segmentado).

Uso:
    python prepare_filtered_data.py
"""

import logging
import os
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_annotation_file(path):
    """Lê um arquivo de anotação e retorna lista de (filename, label)."""
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                entries.append((parts[0], int(parts[1])))
    return entries


def save_annotation_file(entries, path):
    """Salva uma lista de (filename, label) como arquivo de anotação."""
    with open(path, "w") as f:
        for filename, label in entries:
            f.write(f"{filename} {label}\n")
    logger.info(f"Salvo: {path} ({len(entries)} amostras)")


def stratified_split(entries, val_ratio=0.2, seed=42):
    """Divide entries em treino e validação, estratificado por classe.

    Args:
        entries: lista de (filename, label).
        val_ratio: proporção para validação.
        seed: seed para reprodutibilidade.

    Returns:
        Tupla (train_entries, val_entries).
    """
    random.seed(seed)

    # Agrupa por classe
    by_class = {}
    for filename, label in entries:
        by_class.setdefault(label, []).append((filename, label))

    train_entries, val_entries = [], []

    for label in sorted(by_class.keys()):
        items = by_class[label]
        random.shuffle(items)
        split_idx = int(len(items) * (1 - val_ratio))
        train_entries.extend(items[:split_idx])
        val_entries.extend(items[split_idx:])

    # Shuffle final para misturar classes
    random.shuffle(train_entries)
    random.shuffle(val_entries)

    return train_entries, val_entries


def print_distribution(name, entries):
    """Exibe a distribuição de classes de um conjunto de dados."""
    class_names = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    counts = Counter(label for _, label in entries)
    total = len(entries)
    parts = " | ".join(
        f"{class_names.get(k, k)}: {v} ({100*v/total:.1f}%)"
        for k, v in sorted(counts.items())
    )
    logger.info(f"  {name}: {total} amostras — {parts}")


if __name__ == "__main__":
    from config import Config

    base = Path(Config.BASE_PATH)

    logger.info("=" * 60)
    logger.info("PREPARAÇÃO DOS DADOS FILTRADOS")
    logger.info("=" * 60)

    # 1. Coleta todas as imagens segmentadas disponíveis
    seg_train_dir = base / "3A_images_segmented"
    seg_test_dir_1 = base / "3A_test_images_segmented"
    seg_test_dir_2 = base / "3A_test_images_segmented2"

    seg_train_images = {f.name for f in seg_train_dir.iterdir() if f.is_file()} if seg_train_dir.exists() else set()
    seg_test_images = set()
    if seg_test_dir_1.exists():
        seg_test_images |= {f.name for f in seg_test_dir_1.iterdir() if f.is_file()}
    if seg_test_dir_2.exists():
        seg_test_images |= {f.name for f in seg_test_dir_2.iterdir() if f.is_file()}

    logger.info(f"Imagens segmentadas de treino: {len(seg_train_images)}")
    logger.info(f"Imagens segmentadas de teste: {len(seg_test_images)}")

    # 2. Verifica quais existem também na pasta original
    original_dir = base / "3A_images"
    original_images = {f.name for f in original_dir.iterdir() if f.is_file()} if original_dir.exists() else set()

    # Interseção: imagens que existem em AMBAS as versões
    valid_train_images = seg_train_images & original_images
    valid_test_images = seg_test_images & original_images

    logger.info(f"Interseção treino (original ∩ segmentada): {len(valid_train_images)}")
    logger.info(f"Interseção teste (original ∩ segmentada): {len(valid_test_images)}")

    # 3. Filtra anotações originais para conter apenas imagens válidas
    train_txt = base / "train_COVIDx_CT-3A.txt"
    test_txt = base / "test_COVIDx_CT-3A.txt"

    train_all = [(f, l) for f, l in load_annotation_file(train_txt) if f in valid_train_images]
    test_filtered = [(f, l) for f, l in load_annotation_file(test_txt) if f in valid_test_images]

    logger.info(f"\nApós filtro:")
    logger.info(f"  Treino+Val disponíveis: {len(train_all)}")
    logger.info(f"  Teste disponíveis: {len(test_filtered)}")

    # 4. Divide treino em train+val (80/20, estratificado)
    train_filtered, val_filtered = stratified_split(train_all, val_ratio=0.2, seed=42)

    # 5. Exibe distribuição
    logger.info("\nDistribuição final:")
    print_distribution("Treino", train_filtered)
    print_distribution("Validação", val_filtered)
    print_distribution("Teste", test_filtered)

    # 6. Salva arquivos de anotação filtrados
    output_dir = base
    save_annotation_file(train_filtered, output_dir / "train_filtered.txt")
    save_annotation_file(val_filtered, output_dir / "val_filtered.txt")
    save_annotation_file(test_filtered, output_dir / "test_filtered.txt")

    logger.info("\n" + "=" * 60)
    logger.info("Arquivos gerados com sucesso!")
    logger.info(f"  train_filtered.txt: {len(train_filtered)} amostras")
    logger.info(f"  val_filtered.txt:   {len(val_filtered)} amostras")
    logger.info(f"  test_filtered.txt:  {len(test_filtered)} amostras")
    logger.info("=" * 60)
