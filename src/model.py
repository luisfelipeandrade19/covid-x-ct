import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from torchvision.models import (
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    inception_v3, Inception_V3_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

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


class BaseClassifier(pl.LightningModule):
    """Classe Base com o ciclo de treinamento e métricas."""
    def __init__(self, num_classes, learning_rate, lr_decay_factor=0.1, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self._current_stage = 0

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary"),
            torchmetrics.Specificity(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.AUROC(task="binary")
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def unfreeze_stage(self, stage: int):
        if stage <= self._current_stage and self._current_stage > 0:
            return

        for s in range(self._current_stage + 1, stage + 1):
            if s >= len(self.UNFREEZE_STAGES):
                break
            for name, param in self.model.named_parameters():
                if any(block in name for block in self.UNFREEZE_STAGES[s]):
                    param.requires_grad = True

        self._current_stage = min(stage, len(self.UNFREEZE_STAGES) - 1)

    def extract_features(self, x):
        raise NotImplementedError

    def get_cam_target_layer(self):
        raise NotImplementedError

    def forward(self, x):
        features = self.extract_features(x)
        out = self.cbam(features)
        
        # Adaptive pooling para forçar o tamanho correto não importa o tamanho da imagem de entrada
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.train_metrics(probs, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.val_metrics(probs, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }


# =========================================================================
# CLASSES FILHAS (FABRICA DE MODELOS)
# =========================================================================

class DenseNetClassifier(BaseClassifier):
    def __init__(self, model_name="densenet161", num_classes=2, learning_rate=5e-4, **kwargs):
        super().__init__(num_classes, learning_rate, **kwargs)
        self.model_name = model_name
        
        if model_name == "densenet121":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = 1024
        else:
            self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)
            in_features = 2208
            
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.cbam = CBAM(in_planes=in_features, ratio=8)
        self.UNFREEZE_STAGES = [
            [],
            ['denseblock4', 'norm5'],
            ['denseblock3', 'transition3'],
            ['denseblock2', 'transition2'],
            ['denseblock1', 'transition1', 'conv0', 'norm0']
        ]

    def extract_features(self, x):
        features = self.model.features(x)
        return F.relu(features, inplace=True)

    def get_cam_target_layer(self):
        if self.model_name == "densenet121":
            return [self.model.features.denseblock4.denselayer16.conv2]
        return [self.model.features.denseblock4.denselayer24.conv2]


class ResNetClassifier(BaseClassifier):
    def __init__(self, num_classes=2, learning_rate=5e-4, **kwargs):
        super().__init__(num_classes, learning_rate, **kwargs)
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = 2048
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.cbam = CBAM(in_planes=in_features, ratio=8)
        self.UNFREEZE_STAGES = [
            [],
            ['layer4'],
            ['layer3'],
            ['layer2'],
            ['layer1', 'conv1', 'bn1']
        ]

    def extract_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def get_cam_target_layer(self):
        return [self.model.layer4[-1].conv3]


class EfficientNetClassifier(BaseClassifier):
    def __init__(self, num_classes=2, learning_rate=5e-4, **kwargs):
        super().__init__(num_classes, learning_rate, **kwargs)
        self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        in_features = 1408
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.cbam = CBAM(in_planes=in_features, ratio=8)
        self.UNFREEZE_STAGES = [
            [],
            ['features.8'],
            ['features.7', 'features.6'],
            ['features.5', 'features.4'],
            ['features.3', 'features.2', 'features.1', 'features.0']
        ]

    def extract_features(self, x):
        return self.model.features(x)

    def get_cam_target_layer(self):
        return [self.model.features[-1]]


class InceptionClassifier(BaseClassifier):
    def __init__(self, num_classes=2, learning_rate=5e-4, **kwargs):
        super().__init__(num_classes, learning_rate, **kwargs)
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False, transform_input=False)
        in_features = 2048
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.cbam = CBAM(in_planes=in_features, ratio=8)
        self.UNFREEZE_STAGES = [
            [],
            ['Mixed_7'],
            ['Mixed_6'],
            ['Mixed_5'],
            ['Conv2d_1', 'Conv2d_2', 'Conv2d_3', 'Conv2d_4', 'Mixed_5']
        ]

    def extract_features(self, x):
        # Inception V3 Custom Forward (Bypassing Aux Logits & Flatten)
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        return x

    def get_cam_target_layer(self):
        return [self.model.Mixed_7c]


class ConvNeXtClassifier(BaseClassifier):
    def __init__(self, num_classes=2, learning_rate=5e-4, **kwargs):
        super().__init__(num_classes, learning_rate, **kwargs)
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = 768
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.cbam = CBAM(in_planes=in_features, ratio=8)
        self.UNFREEZE_STAGES = [
            [],
            ['features.7'],
            ['features.5'],
            ['features.3'],
            ['features.1', 'features.0']
        ]

    def extract_features(self, x):
        return self.model.features(x)

    def get_cam_target_layer(self):
        return [self.model.features[-1][-1].block[0]]


def get_model_factory(model_name: str, num_classes=2, learning_rate=5e-4):
    """Fábrica de Modelos que retorna a arquitetura correta baseada no nome."""
    name = model_name.lower()
    if name == "densenet161":
        return DenseNetClassifier(model_name="densenet161", num_classes=num_classes, learning_rate=learning_rate)
    elif name == "densenet121":
        return DenseNetClassifier(model_name="densenet121", num_classes=num_classes, learning_rate=learning_rate)
    elif name == "resnet50":
        return ResNetClassifier(num_classes=num_classes, learning_rate=learning_rate)
    elif name == "efficientnet_b2":
        return EfficientNetClassifier(num_classes=num_classes, learning_rate=learning_rate)
    elif name == "inception_v3":
        return InceptionClassifier(num_classes=num_classes, learning_rate=learning_rate)
    elif name == "convnext_tiny":
        return ConvNeXtClassifier(num_classes=num_classes, learning_rate=learning_rate)
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")