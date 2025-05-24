import copy
import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from yucca.functional.utils.kwargs import filter_kwargs
from abc import abstractmethod

# from augmentations.mask import random_mask


class BaseModule(L.LightningModule):
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
        super().__init__()
        # Loss, optimizer and scheduler parameters
        self.learning_rate = learning_rate

        # losses
        self.loss = None

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        assert 0 < cosine_period_ratio <= 1

        # Save params and start training
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        if weights is not None:
            self.load_weights(weights)

        self.model = torch.compile(model, mode=compile_mode) if compile_mode is not None else model

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        print(f"Using optimizer {optimizer.__class__.__name__} with learning rate {self.learning_rate}")

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

    def load_weights(self, weights):
        self.load_state_dict(torch.load(weights, map_location="cpu")["state_dict"], strict=False)

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
        kwargs["strict"] = False
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
