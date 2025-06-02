import copy
import torch
import torch.nn as nn
from asparagus.modules.lightning_modules.base_module import BaseModule
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE


class ClassificationModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        steps_per_epoch: int,
        epochs: int,
        learning_rate: float,
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: str = None,
    ):
        super().__init__(
            model=model,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            weights=weights,
        )
        # losses
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.loss(pred, y.squeeze().long())

        self.log_dict({"train/loss": loss}, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.loss(pred, y.squeeze().long())

        self.log_dict({"val/loss": loss}, on_step=True, on_epoch=True, sync_dist=True)
