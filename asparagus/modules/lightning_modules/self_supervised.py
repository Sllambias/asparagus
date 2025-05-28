import copy
import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from yucca.functional.utils.kwargs import filter_kwargs
from asparagus.modules.lightning_modules.base_module import BaseModule

# from augmentations.mask import random_mask


class SelfSupervisedModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        steps_per_epoch: int,
        epochs: int,
        learning_rate: float,
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        mask_patch_size: int = 4,
        mask_ratio: float = 0.6,
        compile_mode: str = None,
        rec_loss_masked_only: bool = False,
    ):
        super().__init__(
            model=model,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
        )

        # losses
        self._rec_loss_fn = nn.MSELoss(reduction="mean")
        self.rec_loss_masked_only = rec_loss_masked_only

        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

    def training_step(self, batch, batch_idx):
        x, y, mask = batch["image"], batch["label"], batch["mask"]

        pred = self.model(x)
        loss = self.rec_loss(pred, y, mask=mask if self.rec_loss_masked_only else None)

        self.log_dict({"train/loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch["image"], batch["label"], batch["mask"]

        pred = self.model(x)
        loss = self.rec_loss(pred, y, mask=mask if self.rec_loss_masked_only else None)

        self.log_dict({"val/loss": loss}, sync_dist=True)

    def rec_loss(self, pred, y, mask=None):
        """
        Reconstruction MSE loss. If a mask tensor is provided, the loss will only be calculated on masked tokens.
        """
        if mask is not None:
            y[~mask] = 0
            pred[~mask] = 0

        return self._rec_loss_fn(pred, y)


class SelfSupervisedMultiModelModule(SelfSupervisedModule):
    def __init__(
        self,
        model: nn.Module,
        steps_per_epoch: int,
        epochs: int,
        learning_rate: float,
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        patch_size: list | tuple = None,
        mask_patch_size: int = 4,
        mask_ratio: float = 0.6,
        compile_mode: str = None,
        rec_loss_masked_only: bool = False,
    ):
        super().__init__(
            model=model,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
        )
        # Model parameters
        self.patch_size = patch_size

        # losses
        self._rec_loss_fn = nn.MSELoss(reduction="mean")
        self.rec_loss_masked_only = rec_loss_masked_only

        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        y_hat, mask = self._augment_and_forward(x)
        loss = self.rec_loss(y_hat, y, mask=mask if self.rec_loss_masked_only else None)

        self.log_dict({"train/loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        y_hat, mask = self._augment_and_forward(x)
        loss = self.rec_loss(y_hat, y, mask=mask if self.rec_loss_masked_only else None)

        self.log_dict({"val/loss": loss}, sync_dist=True)
