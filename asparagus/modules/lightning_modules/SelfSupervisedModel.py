import copy
from typing import List
import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from yucca.functional.utils.kwargs import filter_kwargs
from augmentations.mask import random_mask
from models import networks


class SelfSupervisedModel(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        steps_per_epoch: int,
        epochs: int,
        learning_rate: float,
        config: dict,
        optimizer: str = "AdamW",
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        input_channels: int = 1,
        output_channels: int = 1,
        patch_size: list | tuple = None,
        mask_patch_size: int = 4,
        mask_ratio: float = 0.6,
        compile_mode: str = None,
        debug_losses: bool = False,
        rec_loss_masked_only: bool = False,
        norm_type: str = None,  # only for mednext
    ):
        super().__init__()
        # Model parameters
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.patch_size = patch_size

        # Loss, optimizer and scheduler parameters
        self.learning_rate = learning_rate

        # losses
        self._rec_loss_fn = nn.MSELoss(reduction=mse_reduction)  # reconstruction
        self.rec_loss_masked_only = rec_loss_masked_only
        self.optimizer = optimizer

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        assert 0 < cosine_period_ratio <= 1

        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

        self.compile_mode = compile_mode

        # Save params and start training
        self.save_hyperparameters()
        self.model = load_model(model)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        assert x.shape[1] == self.input_channels, f"Expected {self.input_channels} input channels but got {x.shape[1]}"
        assert 0 <= x.min() and x.max() <= 1, "Intensities should be normalized to (0, 1)"

        y_hat, mask = self._augment_and_forward(x)

        loss = self.rec_loss(y_hat, y, mask=mask if self.rec_loss_masked_only else None)

        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        assert len(x.shape) == 5
        assert x.shape[1] == self.input_channels, f"Expected {self.input_channels} input channels but got {x.shape[1]}"
        assert 0 <= x.min() and x.max() <= 1, f"Intensities should be normalized to (0, 1), but was {(x.min(), x.max())}"

        y_hat, mask = self._augment_and_forward(x)
        loss = self.rec_loss(y_hat, y, mask=mask if self.rec_loss_masked_only else None)

        self.log_dict({"val/loss": loss})

    def rec_loss(self, y, y_hat, mask=None):
        """
        Reconstruction MSE loss. If a mask tensor is provided, the loss will only be calculated on masked tokens.
        """
        if mask is not None:
            y = y.clone()
            y_hat = y_hat.clone()
            y[~mask] = 0
            y_hat[~mask] = 0

        return self._rec_loss_fn(y, y_hat)

    def _augment_and_forward(self, x):
        with torch.no_grad():
            # x, mask = random_mask(x, self.mask_ratio, self.mask_patch_size)
            mask = x.copy()

        y_hat = self.model(x)

        assert y_hat is not None
        assert y_hat.shape == x.shape, f"Got shape: {y_hat.shape}, expected: {x.shape}"

        return y_hat, mask

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        print(f"Using optimizer {self.optimizer.class__.__name__} with learning rate {self.learning_rate}")

        # cosine_half_period is from max to min
        cosine_half_period = int(self.cosine_period_ratio * self.epochs) - self.warmup_epochs
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_half_period * self.steps_per_epoch)

        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1.0 / 1000,
                total_iters=self.warmup_epochs * self.steps_per_epoch,
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs * self.steps_per_epoch],
            )
        else:
            scheduler = cosine_scheduler

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,  # scheduler is updated after each batch
        }

        return [optimizer], [scheduler_config]

    def load_state_dict(self, state_dict, *args, **kwargs):
        old_params = copy.deepcopy(self.state_dict())
        state_dict = {
            k: v for k, v in state_dict.items() if (k in old_params) and (old_params[k].shape == state_dict[k].shape)
        }
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        rejected_keys_shape = [k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape]
        rejected_keys_data = []

        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(old_params.keys(), old_params.values(), new_params.values()):
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if param_name not in rejected_keys_new and param_name not in rejected_keys_shape:
                    rejected_keys_data.append(param_name)

        print(f"Succesfully transferred weights for {successful}/{successful+unsuccessful} layers")
        print(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )

    @staticmethod
    def load_model(model, input_channels, output_channels, compile_mode):
        print(f"Loading Model: {model.__class__.__name__}")

        model_kwargs = {
            "input_channels": input_channels,
            "output_channels": output_channels,
        }
        model_kwargs = filter_kwargs(model_func, model_kwargs)
        model = model_func(**model_kwargs)

        model = torch.compile(model, mode=compile_mode) if compile_mode is not None else model
        return model
