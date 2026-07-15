from torchvision.models import densenet161, DenseNet161_Weights
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class SimpleClassifier(pl.LightningModule):
    """Classificador de CT pulmonar baseado em DenseNet161 com descongelamento gradual.

    Utiliza transfer learning: a backbone da DenseNet161 é carregada com pesos
    pré-treinados do ImageNet e progressivamente descongelada durante o treino.
    O classificador final é substituído por uma cabeça personalizada com Dropout.

    Atributos:
        UNFREEZE_STAGES: lista de blocos a descongelar em cada fase.
        model: a DenseNet161 com o classificador substituído.
        criterion: função de perda (CrossEntropy com label smoothing).
    """

    # Blocos a descongelar em cada fase (ordem: mais profundo → mais raso)
    UNFREEZE_STAGES = [
        [],                                                  # Fase 0: tudo congelado
        ['denseblock4', 'norm5'],                            # Fase 1: último bloco
        ['denseblock3', 'transition3'],                      # Fase 2: penúltimo bloco
        ['denseblock2', 'transition2'],                      # Fase 3: bloco intermediário
        ['denseblock1', 'transition1', 'conv0', 'norm0']     # Fase 4: tudo descongelado
    ]

    def __init__(self, num_classes, learning_rate, lr_decay_factor=0.1, weight_decay=1e-4):
        """Inicializa o classificador.

        Args:
            num_classes: número de classes de saída (3: Normal, Pneumonia, COVID-19).
            learning_rate: taxa de aprendizado base.
            lr_decay_factor: fator de decaimento do LR para camadas mais profundas.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay

        # Função de perda com pesos inversamente proporcionais à frequência
        # Normal=17922 (53%), Pneumonia=7965 (24%), COVID=7894 (23%)
        # Peso = total / (n_classes * count)   →   penaliza mais erros nas classes menores
        class_weights = torch.tensor([0.63, 1.41, 1.43])
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.1
       )

        # Carrega a DenseNet161 com pesos pré-treinados do ImageNet
        self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        # Congela toda a backbone (será descongelada gradualmente)
        for param in self.model.parameters():
            param.requires_grad = False

        # Substitui o classificador original por uma cabeça personalizada
        in_features = 2208
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),                                          # Dropout para regularização
            nn.Linear(in_features, 256),                              # Camada intermediária
            nn.ReLU(),                                                # Ativação
            nn.Dropout(0.3),                                          # Segundo dropout
            nn.Linear(256, num_classes),                              # Camada de saída
        )

        # Módulo de Atenção (CBAM) inserido após as features da DenseNet161
        self.cbam = CBAM(in_planes=in_features)

        # Controle da fase atual de descongelamento
        self._current_stage = 0

        # Métricas de acurácia para treino e validação
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def unfreeze_stage(self, stage: int):
        """Descongela todos os blocos até a fase indicada (inclusive).

        Percorre as fases desde a atual até a desejada, habilitando
        requires_grad para os parâmetros de cada bloco.

        Args:
            stage: fase alvo de descongelamento (0 a 4).
        """
        # Evita recongelar fases já descongeladas
        if stage <= self._current_stage and self._current_stage > 0:
            return

        # Descongela cada fase entre a atual e a alvo
        for s in range(self._current_stage + 1, stage + 1):
            if s >= len(self.UNFREEZE_STAGES):
                break
            for name, param in self.model.named_parameters():
                if any(block in name for block in self.UNFREEZE_STAGES[s]):
                    param.requires_grad = True

        # Atualiza a fase atual
        self._current_stage = min(stage, len(self.UNFREEZE_STAGES) - 1)

    def forward(self, x):
        """Passa os dados pela DenseNet161 com módulo CBAM embutido."""
        # Extrai features convolucionais da DenseNet
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        
        # Aplica o Módulo de Atenção (ensina o modelo a focar nas patologias e ignorar ossos)
        out = self.cbam(out)
        
        # Faz o pooling global e classificação final
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        
        return out

    def training_step(self, batch, batch_idx):
        """Executa um passo de treino: forward, cálculo de loss e métricas.

        Args:
            batch: tupla (imagens, rótulos).
            batch_idx: índice do lote atual.

        Returns:
            Valor da loss para o otimizador.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Atualiza a métrica de acurácia de treino
        self.train_acc(torch.argmax(logits, dim=1), y)

        # Registra loss e acurácia no logger
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Executa um passo de validação: forward, cálculo de loss e métricas.

        Args:
            batch: tupla (imagens, rótulos).
            batch_idx: índice do lote atual.

        Returns:
            Valor da loss de validação.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Atualiza a métrica de acurácia de validação
        self.val_acc(torch.argmax(logits, dim=1), y)

        # Registra loss e acurácia no logger
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configura o otimizador AdamW e o scheduler ReduceLROnPlateau.

        Returns:
            Dicionário com otimizador e scheduler monitorando val_loss.
        """
        param_groups = self._build_param_groups()
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _build_param_groups(self):
        """Monta param groups com LR decrescente para camadas mais profundas.

        Camadas descongeladas mais recentemente recebem o LR base;
        camadas descongeladas em fases anteriores recebem LRs
        progressivamente menores (multiplicadas por lr_decay_factor).

        Returns:
            Lista de dicionários com 'params' e 'lr' para o otimizador.
        """
        
        # Agrupa parâmetros por fase de descongelamento
        stage_params = {s: [] for s in range(len(self.UNFREEZE_STAGES))}
        classifier_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Parâmetros do classificador e CBAM vão em grupo separado
            if 'classifier' in name or 'cbam' in name:
                classifier_params.append(param)
                continue

            # Identifica a qual fase o parâmetro pertence
            assigned = False
            for s, blocks in enumerate(self.UNFREEZE_STAGES):
                if any(block in name for block in blocks):
                    stage_params[s].append(param)
                    assigned = True
                    break
            if not assigned:
                classifier_params.append(param)

        groups = []

        # Classificador sempre recebe o LR completo
        if classifier_params:
            groups.append({'params': classifier_params, 'lr': self.learning_rate})

        # Fases mais antigas recebem LR progressivamente menor
        for s in range(self._current_stage, 0, -1):
            if stage_params[s]:
                distance = self._current_stage - s + 1
                lr = self.learning_rate * (self.lr_decay_factor ** distance)
                groups.append({'params': stage_params[s], 'lr': lr})

        # Fallback: se nenhum parâmetro treinável, cria um dummy
        return groups if groups else [{'params': [torch.zeros(1, requires_grad=True)], 'lr': self.learning_rate}]