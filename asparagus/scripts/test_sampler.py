"""
See e.g. sample "0" which occurs in very different orders and on both devices.
"""

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, Dataset, RandomSampler

torch.manual_seed(420)


class RandomDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(10).float()
        self.name = torch.arange(10).float()

    def __getitem__(self, idx):
        return self.name[idx], self.data[idx]

    def __len__(self):
        return len(self.data)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: str):
        if stage == "fit":
            self.setup_fit()

    def setup_fit(self):
        self.train_dataset = RandomDataset()
        self.val_dataset = RandomDataset()

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, num_samples=100, replacement=True)
        if dist.is_initialized():
            sampler = DistributedSamplerWrapper(sampler)
        dl = DataLoader(
            self.train_dataset,
            num_workers=0,
            batch_size=5,
            pin_memory=False,
            drop_last=True,
            sampler=sampler,
        )
        return dl


class SomeLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.tensor(0.0))
        self.p2 = torch.nn.Parameter(torch.tensor(0.0))

    def training_step(self, batch):
        x, y = batch
        print(
            "epoch:",
            self.current_epoch,
            "rank:",
            self.local_rank,
            "samples:",
            x,
        )
        return F.mse_loss(x * self.p1 + self.p2, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
        )

        return {
            "optimizer": optimizer,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Use DDP training")
    args = parser.parse_args()
    DDP = args.ddp
    devices = 2

    lightning_module = SomeLightningModule()
    datamodule = PretrainDataModule()

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=6,
        limit_train_batches=5,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        devices=1,
    )

    trainer2 = pl.Trainer(
        strategy="ddp_notebook",
        accelerator="cpu",
        max_epochs=6 // devices,
        limit_train_batches=5,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        devices=2,
    )

    if DDP:
        print("STARTING DDP TRAINING................")
        trainer2.fit(lightning_module, datamodule=datamodule)
    else:
        print("STARTING NON DDP TRAINING................")
        trainer.fit(lightning_module, datamodule=datamodule)
