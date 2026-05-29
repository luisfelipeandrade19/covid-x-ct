import pytorch_lightning as pl


class GradualUnfreezing(pl.Callback):
    def __init__(self, epochs_per_stage: int = 5, max_stage: int = 4):
        super().__init__()
        self.epochs_per_stage = epochs_per_stage
        self.max_stage = max_stage
        self._last_stage = 0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_epoch = trainer.current_epoch
        target_stage = min(current_epoch // self.epochs_per_stage, self.max_stage)

        if target_stage > self._last_stage:
            pl_module.unfreeze_stage(target_stage)
            self._last_stage = target_stage

            # Recria o optimizer com os novos param groups
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

            stage_blocks = pl_module.UNFREEZE_STAGES[target_stage]
            print(
                f"\n>>> [GradualUnfreezing] Época {current_epoch}: "
                f"descongelando fase {target_stage} — {stage_blocks}"
            )
