from torchvision.models import densenet161, DenseNet161_Weights
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SimpleClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'denseblock4' in name or 'norm5' in name:
                param.requires_grad = True

        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        # Métricas
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}