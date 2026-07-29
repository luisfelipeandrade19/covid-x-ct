import logging

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class GradualUnfreezing(pl.Callback):
    """Callback para descongelamento gradual da backbone durante o treino.

    A cada N épocas (epochs_per_stage), descongela a próxima fase da backbone,
    permitindo que camadas mais profundas sejam ajustadas progressivamente.
    Também reconfigura o otimizador e scheduler a cada transição de fase.

    Args:
        epochs_per_stage: número de épocas entre cada fase de descongelamento.
        max_stage: fase máxima a ser atingida (corresponde ao UNFREEZE_STAGES do modelo).
    """

    def __init__(self, epochs_per_stage: int = 5, max_stage: int = 4):
        super().__init__()
        self.epochs_per_stage = epochs_per_stage
        self.max_stage = max_stage
        self._last_stage = 0   # Última fase aplicada

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Executado no início de cada época de treino.

        Verifica se é hora de avançar para a próxima fase de descongelamento.
        Se sim, descongela os blocos correspondentes e recria o otimizador
        com os novos param groups (LR diferenciado por camada).

        Args:
            trainer: instância do Trainer do Lightning.
            pl_module: modelo sendo treinado (SimpleClassifier).
        """
        current_epoch = trainer.current_epoch

        # Calcula a fase alvo com base na época atual
        target_stage = min(current_epoch // self.epochs_per_stage, self.max_stage)

        # Só avança se a fase alvo for maior que a última aplicada
        if target_stage > self._last_stage:
            # Descongela os blocos da nova fase
            pl_module.unfreeze_stage(target_stage)
            self._last_stage = target_stage

            # Reconfigura o otimizador com os novos param groups
            new_optim_config = pl_module.configure_optimizers()
            trainer.optimizers = [new_optim_config["optimizer"]]
            trainer.lr_schedulers = [
                {
                    "scheduler": new_optim_config["lr_scheduler"]["scheduler"],
                    "monitor": new_optim_config["lr_scheduler"].get("monitor", "val_loss"),
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": True,
                }
            ]

            # Log da transição de fase
            stage_blocks = pl_module.UNFREEZE_STAGES[target_stage]
            logger.info(
                f"Época {current_epoch}: "
                f"descongelando fase {target_stage} — {stage_blocks}"
            )

