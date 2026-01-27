import lightning as pl
import logging
import torch.distributed as dist
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import ClsRegDataset, ClsRegTestDataset, SegDataset, SegTestDataset
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose
from typing import Literal, Optional


class SegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        test_samples: list = [],
        train_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.test_samples = test_samples
        self.val_split = val_split

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        self.train_dataset = SegDataset(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = SegDataset(
            self.val_split,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = SegTestDataset(
            self.test_samples,
            transforms=self.test_transforms,
        )

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=collate_return,
        )


class ClsRegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        test_samples: Optional[list] = [],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_samples = test_samples
        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        self.train_dataset = ClsRegDataset(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = ClsRegDataset(
            self.val_split,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = ClsRegTestDataset(
            self.test_samples,
            transforms=self.test_transforms,
        )

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
            num_workers=self.num_workers // 2,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            persistent_workers=True,
            drop_last=True,
            sampler=sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=1,
            batch_size=1,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=collate_return,
        )


if __name__ == "__main__":
    from asparagus.functional.loading import load_json

    dataset_json = load_json(
        "/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task997_LauritSynSeg/dataset.json"
    )
    splits = load_json(
        "/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task997_LauritSynSeg/split_80_20.json"
    )[0]
    train_split = splits["train"]
    val_split = splits["val"]
    data_module = SegDataModule(
        train_split=train_split,
        val_split=val_split,
        batch_size=2,
        num_workers=6,
    )
    data_module.setup("fit")
    data_module_iterator = iter(data_module.train_dataloader())
    x = next(data_module_iterator)
    print(type(x))
    print(next(iter(data_module.train_dataset))["image"].shape)
