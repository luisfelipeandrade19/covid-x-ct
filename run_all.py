"""Executa todos os scripts de avaliação e visualização em sequência.

Gera todos os gráficos de uma só vez na pasta 'outputs/':
    - evaluate:     confusion_matrix, training_curves, ROC, Precision-Recall
    - calibration:  Reliability Diagrams (antes/depois) + Temperature Scaling
    - tta:          Comparação de matrizes de confusão (Normal vs TTA)
    - visualize:    Grad-CAM e Grad-CAM++ grids

Uso:
    python run_all.py
"""

import logging
import time

from evaluate import main as evaluate_main
from calibration import main as calibration_main
from tta import main as tta_main
from visualize import main as visualize_main

# Configura o logging para todo o projeto
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Executa os 4 módulos de geração de gráficos em sequência."""
    start = time.time()

    logger.info("=" * 60)
    logger.info("ETAPA 1/4 — Avaliação (evaluate.py)")
    logger.info("=" * 60)
    evaluate_main()

    logger.info("")
    logger.info("=" * 60)
    logger.info("ETAPA 2/4 — Calibração (calibration.py)")
    logger.info("=" * 60)
    calibration_main()

    logger.info("")
    logger.info("=" * 60)
    logger.info("ETAPA 3/4 — Test-Time Augmentation (tta.py)")
    logger.info("=" * 60)
    tta_main()

    logger.info("")
    logger.info("=" * 60)
    logger.info("ETAPA 4/4 — Visualização Grad-CAM (visualize.py)")
    logger.info("=" * 60)
    visualize_main()

    elapsed = time.time() - start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Todos os gráficos gerados com sucesso! Tempo total: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
