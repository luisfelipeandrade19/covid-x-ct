import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from config import Config
from loaders import test_loader, val_loader
from model import SimpleClassifier

class ECELoss(nn.Module):
    """ Calcula o Expected Calibration Error (ECE) com 15 faixas (bins). """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()

        # Cria faixas bins de 0.0 a 1.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundariesp[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin. item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        returnece.item()

class ModelWithTemperature(nn.Module):
    """ Wrapper para um modelo PyTorch que aplica Temperature Scaling. """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # A temperatura começa em 1.5 (Um chute inicial) e será otimizada
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self,x):
        logits = self.model(x)
        # Ao dividir os logits pela temperatura antes do softmax, alteramos a calibração
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        """ Otimiza a temperatura T usando o conjunto de Validação """
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()

        # Coletamos todos os logits do conjuntod de validação sem alterar a reversed
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                logits = self.model(x.to(device))
                logits_list.append(logits)
                labels_list.append(y.to(device))

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Calculamos as perdas antes do ajuste
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels)
        logger.info(f"Antes da Calibração (Validação) - NLL: {before_temperature_nll:.3f}, ECE:{before_temperature_ece:.3f}")

        # Otimizamos a Temperatura T minimizando a NLL
        # Usamos o algoritmo L-BFGS, que converge super rápido para otimização escalar
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Resultados depois do ajuste

        after_temperature_nll = nll_criterion(logits / self.temperature, labels).item()
        after_temperature_ece = ece_criterion(logits / self.temperature, labels)
        logger.info(f"Temperatura ideal encontrada: {self.temperature.item():.3f}")
        logger.info(f"Depois da Calibração (Validação) - NLL : {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}")

def plot_reliability_diagram(logits, labels, title, filename, n_bins=10):
    """ Gera um gráfico comparando confiança predita com acurácia real """
    softmaxes = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    bin_boundaries = np.linscape(0,1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies, bin_confidences = [], []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.any():
            bin_accuracies.append(accuracies[in_bin].float().mean().item())
            bin_confidences.append(confidences[in_bin].mean().item())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    plt.figure(figsize=(6, 6))

    plt.bar(bin_confidences, bin_accuracies, width=0.1, color='blue', alpha=0.5, label='Modelo')

    # Plota linha vermelha que represanta a 'Perfeição'
    plt.plot([0, 1], [0, 1], 'r--', label='Calibração Perfeita')

    plt.xlabel('Confiança')
    plt.ylabel('Acurácia')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    caminho_outputs = Path(Config.IMG_OUTPUTS_PATH)
    caminho_outputs.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho_outputs / filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carrega o modelo normal
    checkpoint_path = os.path.join(Config.BASE_PATH, "checkpoints", "best_model.ckpt")
    model = SimpleClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # 2. Coleta logits puros do conjunto de TESTE para avaliar o "Antes"
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            test_logits_list.append(logits)
            test_labels_list.append(y.to(device))
            
    test_logits = torch.cat(test_logits_list)
    test_labels = torch.cat(test_labels_list)

    # 3. Avaliação no Teste (Sem Calibração)
    ece_criterion = ECELoss()
    ece_before = ece_criterion(test_logits, test_labels)
    logger.info(f"\nConjunto de TESTE - ECE ANTES da calibração: {ece_before:.3f}")
    plot_reliability_diagram(test_logits, test_labels, 
                             f"Reliability Diagram (Antes) - ECE: {ece_before:.3f}", 
                             "reliability_diagram_before.png")

    # 4. Aplica o Temperature Scaling
    # Passamos o modelo original para o Wrapper
    calibrated_model = ModelWithTemperature(model)
    calibrated_model.to(device)
    
    logger.info("\n--- Otimizando Temperatura ---")
    # A mágica acontece aqui: achamos o T no val_loader
    calibrated_model.set_temperature(val_loader, device)

    # 5. Avaliação no Teste (Com Calibração)
    # Aplicamos a temperatura T nos logits de teste
    with torch.no_grad():
        scaled_test_logits = calibrated_model(test_logits)

    ece_after = ece_criterion(scaled_test_logits, test_labels)
    logger.info(f"\nConjunto de TESTE - ECE DEPOIS da calibração: {ece_after:.3f}")
    plot_reliability_diagram(scaled_test_logits, test_labels, 
                             f"Reliability Diagram (Depois) - ECE: {ece_after:.3f}", 
                             "reliability_diagram_after.png")
