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
    """Lê um arquivo de anotação e retorna lista de (filename, label).
    
    Remove a classe 0 (Normal) e remapeia as classes doentes:
        1 (Pneumonia) -> 0
        2 (COVID-19) -> 1
    """
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[1])
                
                # Ignora a classe Normal (0)
                if label == 0:
                    continue
                    
                # Remapeia Pneumonia(1)->0 e COVID-19(2)->1
                new_label = label - 1
                entries.append((parts[0], new_label))
    return entries


def save_annotation_file(entries, path):
    """Salva uma lista de (filename, label) como arquivo de anotação."""
    with open(path, "w") as f:
        for filename, label in entries:
            f.write(f"{filename} {label}\n")
    logger.info(f"Salvo: {path} ({len(entries)} amostras)")


import re

def get_patient_id(filename):
    """Extrai o identificador único do paciente a partir do nome do arquivo.
    
    Usa heurísticas baseadas nos padrões de nomenclatura do COVIDx CT-3A:
    - 137covid_patient100_SR_2_IM00028.png -> 137covid_patient100
    - volume-covid19-A-0698_ct-0017.png -> volume-covid19-A-0698
    - CP_10_01.png -> CP_10
    """
    name = filename.split('.')[0] # Remove extensão
    
    # Padrão: 137covid_patient100...
    match = re.search(r'(.*patient\d+)', name)
    if match: return match.group(1)
        
    # Padrão: volume-covid19-A-0698...
    match = re.search(r'(volume-covid19-[A-Za-z0-9]+-\d+)', name)
    if match: return match.group(1)
    
    # Padrão CP: CP_10_01 -> CP_10
    if name.startswith('CP_'):
        parts = name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
            
    # Fallback: remove a última parte (que costuma ser o número do slice)
    parts = name.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    
    parts = name.split('-')
    if len(parts) > 1:
        return '-'.join(parts[:-1])
        
    return name

def stratified_patient_split(entries, val_ratio=0.2, seed=42):
    """Divide entries em treino e validação a NÍVEL DE PACIENTE.
    
    Garante que todos os slices de um mesmo paciente fiquem exclusivamente
    no treino ou na validação, evitando data leakage. A estratificação é 
    feita com base na classe do paciente.

    Args:
        entries: lista de (filename, label).
        val_ratio: proporção para validação.
        seed: seed para reprodutibilidade.

    Returns:
        Tupla (train_entries, val_entries).
    """
    random.seed(seed)

    # 1. Agrupa slices por paciente
    # patients = { patient_id: {'label': class, 'slices': [(fn, label), ...]} }
    patients = {}
    for filename, label in entries:
        pid = get_patient_id(filename)
        if pid not in patients:
            patients[pid] = {'label': label, 'slices': []}
        patients[pid]['slices'].append((filename, label))

    # 2. Agrupa pacientes por classe para estratificação
    by_class = {}
    for pid, data in patients.items():
        by_class.setdefault(data['label'], []).append(pid)

    train_entries, val_entries = [], []
    train_pids, val_pids = set(), set()

    # 3. Divide os pacientes de cada classe
    for label in sorted(by_class.keys()):
        pids = by_class[label]
        random.shuffle(pids)
        split_idx = int(len(pids) * (1 - val_ratio))
        
        # Adiciona pacientes ao treino
        for pid in pids[:split_idx]:
            train_entries.extend(patients[pid]['slices'])
            train_pids.add(pid)
            
        # Adiciona pacientes à validação
        for pid in pids[split_idx:]:
            val_entries.extend(patients[pid]['slices'])
            val_pids.add(pid)

    # Verifica leakage
    overlap = train_pids & val_pids
    if overlap:
        logger.error(f"FATAL LEAKAGE! {len(overlap)} pacientes em ambos os splits.")
    else:
        logger.info(f"Split patient-disjoint com sucesso. Treino: {len(train_pids)} pctes | Val: {len(val_pids)} pctes.")

    # Shuffle final para misturar os slices durante o carregamento
    random.shuffle(train_entries)
    random.shuffle(val_entries)

    return train_entries, val_entries


def print_distribution(name, entries):
    """Exibe a distribuição de classes de um conjunto de dados."""
    class_names = {0: "Pneumonia", 1: "COVID-19"}
    counts = Counter(label for _, label in entries)
    total = len(entries)
    parts = " | ".join(
        f"{class_names.get(k, k)}: {v} ({100*v/total:.1f}%)"
        for k, v in sorted(counts.items())
    )
    logger.info(f"  {name}: {total} amostras — {parts}")


if __name__ == "__main__":
    import sys
    import os
    
    # Força a raiz do projeto ser a prioridade número 1 nos imports
    # Resolve o erro "dataset is not a package"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from src.config import Config

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
    
    # Une as duas listas de imagens que passaram na interseção
    valid_files = valid_train_images | valid_test_images

    def map_and_filter(path, label_str, valid_set):
        return [(f, l) for f, l in load_annotation_file(path) if f in valid_set]

    train_orig = base / "train_COVIDx_CT-3A.txt"
    test_orig = base / "test_COVIDx_CT-3A.txt"

    train_all = map_and_filter(train_orig, "treino", valid_files)
    test_all = map_and_filter(test_orig, "teste", valid_files)

    logger.info("--- Resumo ---")
    
    all_entries = train_all + test_all
    logger.info(f"  Total de amostras disponíveis: {len(all_entries)}")
    
    train_filtered, temp_filtered = stratified_patient_split(all_entries, val_ratio=0.2, seed=42)
    val_filtered, test_filtered = stratified_patient_split(temp_filtered, val_ratio=0.5, seed=42)

    logger.info("\nDistribuição final:")
    def log_dist(name, entries):
        counts = Counter([label for _, label in entries])
        pids = set(get_patient_id(fn) for fn, _ in entries)
        logger.info(f"  {name}: {len(entries)} slices | {len(pids)} pacientes")
        logger.info(f"    Pneumonia: {counts.get(0, 0)} | COVID-19: {counts.get(1, 0)}")

    log_dist("Treino", train_filtered)
    log_dist("Validação", val_filtered)
    log_dist("Teste", test_filtered)

    save_annotation_file(train_filtered, base / "train_filtered.txt")
    save_annotation_file(val_filtered, base / "val_filtered.txt")
    save_annotation_file(test_filtered, base / "test_filtered.txt")

    logger.info("\nProcesso concluído com sucesso!")
    logger.info("=" * 60)
