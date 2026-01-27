import logging
import torch
from gardening_tools.modules.losses.DiceCE import DiceCE
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_warn
from torch import Tensor, nn
from torchmetrics.segmentation import DiceScore
from typing import Sequence, Tuple, Union


class OnlineSegmentationPlugin(Callback):
    def __init__(
        self,
        data_module,
        model: nn.Module,
        output_channels: int,
        epochs: int = 3,
        batch_size: int = 2,
        every_n_epochs: int = 5,
        train_n_last_params: int = 6,
        limit_train_batches: int = 250,
        limit_val_batches: int = 50,
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.data_module = data_module
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.output_channels = output_channels
        self.model = model
        self.train_n_last_params = train_n_last_params
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        if train_n_last_params % 2 != 0:
            logging.warn("Train_n_last_layers not a multiple of 2. Most layers are weight+bias")

    def setup(self, trainer, pl_module, stage="fit"):
        self.model = self.model.to(pl_module.device)
        self.data_module.setup("fit")
        self.loss = DiceCE()
        self.dice = DiceScore(self.output_channels, include_background=False, average="macro").to(pl_module.device)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.current_epoch % self.every_n_epochs == 0:
            return

        self.model.apply(weight_reset)
        self.model.load_state_dict(pl_module.state_dict().copy(), strict=False)

        total_n_params = len(list(self.model.named_parameters()))
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if idx < total_n_params - self.train_n_last_params:
                param.requires_grad = False

        accel = trainer.accelerator_connector if hasattr(trainer, "accelerator_connector") else trainer._accelerator_connector
        if accel.is_distributed:
            if accel.use_ddp:
                from torch.nn.parallel import DistributedDataParallel

                self.model = DistributedDataParallel(self.model, device_ids=[pl_module.device])
            elif accel.use_dp:
                from torch.nn.parallel import DataParallel

                self.model = DataParallel(self.model, device_ids=[pl_module.device])
            else:
                rank_zero_warn("Does not support this type of distributed accelerator. The online evaluator will not sync.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.train(pl_module)

    def transfer_batch_to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        batch["image"] = batch["image"].to(device)
        batch["label"] = batch["label"].to(device)
        return batch

    def train_step(
        self,
        batch,
        pl_module: LightningModule,
    ):
        self.model.train()
        with torch.no_grad():
            batch = self.transfer_batch_to_device(batch, pl_module.device)
            batch = self.on_after_batch_transfer(batch)
            x, y = batch["image"], batch["label"]

        pred = self.model(x)

        loss = self.loss(pred, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("online_train_loss", loss.detach(), prog_bar=False)

        """
        Calculating the accuracy currently leads to a memory leak. Tried .detach() after and .detach on 
        the pred and "with torch.no_grad():".
        It is a known issue with some torchmetrics.

        acc = self.dice(pred.argmax(1), y.squeeze(1).long())
        pl_module.log("online_train_acc", acc, prog_bar=False)
        """

    def val_step(
        self,
        batch,
        pl_module: LightningModule,
    ):
        self.model.eval()
        with torch.no_grad():
            batch = self.transfer_batch_to_device(batch, pl_module.device)
            batch = self.on_after_batch_transfer(batch)
            x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.loss(pred, y)

        pl_module.log("online_seg_val_loss", loss.detach(), prog_bar=False)

        """
        see train_step 
        
        acc = self.dice(pred.argmax(1), y.squeeze(1).long())
        pl_module.log("online_seg_val_acc", acc.item(), prog_bar=False)
        """

    def train(self, pl_module) -> None:
        for epoch in range(self.epochs):
            for idx, batch in enumerate(self.data_module.train_dataloader()):
                if idx >= self.limit_train_batches:
                    break
                self.train_step(batch, pl_module)
            # for idx, batch in enumerate(self.data_module.val_dataloader()):
            #    if idx >= self.limit_val_batches:
            #        break
            #    self.val_step(batch, pl_module)

    def state_dict(self) -> dict:
        return {"state_dict": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_after_batch_transfer(self, batch):
        # implement transforms here
        return batch


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
