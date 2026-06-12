from torchvision.models import densenet161, DenseNet161_Weights
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

    def __init__(self, num_classes, learning_rate, lr_decay_factor=0.1):
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

        # Função de perda com label smoothing para regularização
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Carrega a DenseNet161 com pesos pré-treinados do ImageNet
        self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        # Congela toda a backbone (será descongelada gradualmente)
        for param in self.model.parameters():
            param.requires_grad = False

        # Substitui o classificador original por uma cabeça personalizada
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),                                          # Dropout para regularização
            nn.Linear(self.model.classifier.in_features, 512),        # Camada intermediária
            nn.ReLU(),                                                # Ativação
            nn.Dropout(0.2),                                          # Segundo dropout
            nn.Linear(512, num_classes),                              # Camada de saída
        )

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
        """Passa os dados pela DenseNet161."""
        return self.model(x)

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
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)
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

            # Parâmetros do classificador vão em grupo separado
            if 'classifier' in name:
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