"""Script de verificação dos dados segmentados disponíveis.

Mapeia pastas e arquivos de anotação segmentados para planejar o experimento.

Uso:
    python check_segmented.py
"""

import os
from pathlib import Path
from src.config import Config

base = Path(Config.BASE_PATH)

print("=" * 60)
print("VERIFICAÇÃO DOS DADOS SEGMENTADOS")
print(f"BASE_PATH: {base}")
print("=" * 60)

# 1. Lista TODOS os arquivos .txt na raiz (anotações)
print("\n📄 Arquivos de anotação encontrados:")
for f in sorted(base.glob("*.txt")):
    lines = sum(1 for _ in open(f))
    print(f"   {f.name} ({lines} linhas)")

# 2. Lista TODAS as pastas que contêm imagens
print("\n📁 Pastas de imagens encontradas:")
for d in sorted(base.iterdir()):
    if d.is_dir() and "image" in d.name.lower():
        count = sum(1 for _ in d.iterdir() if _.is_file())
        print(f"   {d.name}/ ({count} imagens)")

# 3. Pastas com "segment" no nome
print("\n🔍 Pastas/arquivos com 'segment' no nome:")
for item in sorted(base.rglob("*segment*")):
    if item.is_dir():
        count = sum(1 for _ in item.iterdir() if _.is_file())
        print(f"   [DIR]  {item.relative_to(base)}/ ({count} arquivos)")
    elif item.is_file() and item.parent == base:
        lines = sum(1 for _ in open(item))
        print(f"   [FILE] {item.name} ({lines} linhas)")

# 4. Verifica cobertura: imagens segmentadas vs anotações originais
print("\n📊 Análise de cobertura:")
seg_dirs = [d for d in base.iterdir() if d.is_dir() and "segment" in d.name.lower()]

if seg_dirs:
    # Junta todas as imagens segmentadas em um set
    all_seg_images = set()
    for d in seg_dirs:
        for f in d.iterdir():
            if f.is_file():
                all_seg_images.add(f.name)
    print(f"   Total de imagens segmentadas (todas as pastas): {len(all_seg_images)}")

    # Cruza com cada arquivo de anotação original
    for txt in sorted(base.glob("*.txt")):
        if "segment" in txt.name:
            continue
        with open(txt) as f:
            entries = [line.strip().split()[0] for line in f if line.strip()]
        found = sum(1 for e in entries if e in all_seg_images)
        print(f"   {txt.name}: {found}/{len(entries)} imagens com versão segmentada ({100*found/len(entries):.1f}%)")
else:
    print("   Nenhuma pasta segmentada encontrada.")

print("\n" + "=" * 60)
