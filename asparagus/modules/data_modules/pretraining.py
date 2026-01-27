import lightning as pl
import logging
import torch.distributed as dist
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.transforms.presets import pretrain_CPU_train_transforms, pretrain_CPU_val_transforms
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose
from typing import Literal, Optional


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        train_transforms: Optional[Compose] = pretrain_CPU_train_transforms,
        val_transforms: Optional[Compose] = pretrain_CPU_val_transforms,
        num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.num_samples = num_samples

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        self.train_dataset = PretrainDataset(self.train_split, transforms=self.train_transforms)
        self.val_dataset = PretrainDataset(self.val_split, transforms=self.val_transforms)

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, num_samples=999999, replacement=True)
        if dist.is_initialized():
            sampler = DistributedSamplerWrapper(sampler)

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        sampler = RandomSampler(self.val_dataset, num_samples=999999, replacement=True)
        if dist.is_initialized():
            sampler = DistributedSamplerWrapper(sampler)

        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            persistent_workers=True,
            drop_last=True,
            sampler=sampler,
        )


if __name__ == "__main__":
    from asparagus.functional.loading import load_json

    splits = load_json("/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task999_DummyData/splits.json")
    train_split = splits["train"]
    val_split = splits["validation"]
    data_module = PretrainDataModule(
        train_split=train_split,
        val_split=val_split,
        batch_size=2,
        num_workers=6,
    )
    data_module.setup("fit")
    print(data_module.train_dataset[0]["image"].shape)
