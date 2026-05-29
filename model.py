from torchvision.models import densenet161, DenseNet161_Weights
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SimpleClassifier(pl.LightningModule):

    # Blocos a descongelar em cada fase (ordem: mais profundo → mais raso)
    UNFREEZE_STAGES = [
        [],                                                  
        ['denseblock4', 'norm5'],                            
        ['denseblock3', 'transition3'],                      
        ['denseblock2', 'transition2'],                      
        ['denseblock1', 'transition1', 'conv0', 'norm0']
    ]

    def __init__(self, num_classes, learning_rate, lr_decay_factor=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        # Congela toda a backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Substitui o classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        self._current_stage = 0

        # Métricas
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    # ------------------------------------------------------------------ #
    #  Descongelamento gradual                                            #
    # ------------------------------------------------------------------ #
    def unfreeze_stage(self, stage: int):
        """Descongela todos os blocos até a fase indicada (inclusive)."""
        if stage <= self._current_stage and self._current_stage > 0:
            return  # já descongelado

        for s in range(self._current_stage + 1, stage + 1):
            if s >= len(self.UNFREEZE_STAGES):
                break
            for name, param in self.model.named_parameters():
                if any(block in name for block in self.UNFREEZE_STAGES[s]):
                    param.requires_grad = True

        self._current_stage = min(stage, len(self.UNFREEZE_STAGES) - 1)

    # ------------------------------------------------------------------ #
    #  Forward / steps                                                     #
    # ------------------------------------------------------------------ #
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(torch.argmax(logits, dim=1), y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(torch.argmax(logits, dim=1), y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------ #
    #  Optimizer com learning rates discriminativos                        #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        param_groups = self._build_param_groups()
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _build_param_groups(self):
        """Monta param groups com LR decrescente para camadas mais profundas.

        Camadas descongeladas mais recentemente recebem o LR base;
        camadas descongeladas em fases anteriores recebem LRs
        progressivamente menores (multiplicadas por lr_decay_factor).
        """
        # Agrupa parâmetros treináveis por fase
        stage_params = {s: [] for s in range(len(self.UNFREEZE_STAGES))}
        classifier_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Classifier (head) — sempre no grupo de maior LR
            if 'classifier' in name:
                classifier_params.append(param)
                continue

            # Descobre a qual fase o parâmetro pertence
            assigned = False
            for s, blocks in enumerate(self.UNFREEZE_STAGES):
                if any(block in name for block in blocks):
                    stage_params[s].append(param)
                    assigned = True
                    break
            if not assigned:
                classifier_params.append(param)

        groups = []

        # Classifier sempre com LR cheio
        if classifier_params:
            groups.append({'params': classifier_params, 'lr': self.learning_rate})

        # Backbone: fases mais recentes → LR mais alto
        # Fase current_stage recebe lr * decay^1, current_stage-1 recebe lr * decay^2, ...
        for s in range(self._current_stage, 0, -1):
            if stage_params[s]:
                distance = self._current_stage - s + 1
                lr = self.learning_rate * (self.lr_decay_factor ** distance)
                groups.append({'params': stage_params[s], 'lr': lr})

        return groups if groups else [{'params': [torch.zeros(1, requires_grad=True)], 'lr': self.learning_rate}]