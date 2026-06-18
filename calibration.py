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

        