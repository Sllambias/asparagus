import lightning as pl
import logging
import polars
import torch
import torch.distributed as dist
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import (
    ClsRegDataset,
    ClsRegTestDataset,
    SegDataset,
    SegTestDataset,
    SingleSubjectPredictDataset,
    TableClsRegDataset,
    TableClsRegTestDataset,
)
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose
from typing import Literal, Optional


class TrainingDataModule(pl.LightningDataModule):
    """Setup shared by the segmentation and classification/regression data modules."""

    dataset_class = None
    test_dataset_class = None

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = self.dataset_class(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = self.dataset_class(
            self.val_split,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = self.test_dataset_class(
            self.test_samples,
            transforms=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = SingleSubjectPredictDataset(
            self.predict_samples,
            transforms=self.predict_transforms,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=collate_return,
        )


class SegDataModule(TrainingDataModule):
    dataset_class = SegDataset
    test_dataset_class = SegTestDataset

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        test_samples: list = [],
        predict_samples: Optional[list] = [],
        predict_transforms: Optional[Compose] = None,
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
        self.predict_samples = predict_samples
        self.predict_transforms = predict_transforms

        logging.info(f"Using {self.num_workers} workers")

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, num_samples=999999, replacement=True)
        if dist.is_initialized():
            sampler = DistributedSamplerWrapper(sampler)

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
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
            pin_memory=torch.cuda.is_available(),
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
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            collate_fn=collate_return,
        )


class ClsRegDataModule(TrainingDataModule):
    dataset_class = ClsRegDataset
    test_dataset_class = ClsRegTestDataset

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
        test_samples: Optional[list] = [],
        predict_samples: Optional[list] = [],
        use_random_datasampler: Optional[bool] = True,
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
        self.use_random_datasampler = use_random_datasampler
        self.predict_samples = predict_samples
        self.predict_transforms = predict_transforms
        logging.info(f"Using {self.num_workers} workers")

    def train_dataloader(self):
        sampler = None
        if self.use_random_datasampler:
            sampler = RandomSampler(self.train_dataset, num_samples=999999, replacement=True)
            sampler = DistributedSamplerWrapper(sampler) if dist.is_initialized() else sampler

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            drop_last=True,
            shuffle=sampler is None,
            sampler=sampler,
        )

    def val_dataloader(self):
        sampler = None
        if self.use_random_datasampler:
            sampler = RandomSampler(self.val_dataset, num_samples=999999, replacement=True)
            sampler = DistributedSamplerWrapper(sampler) if dist.is_initialized() else sampler

        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers // 2,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            drop_last=False,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=1,
            batch_size=1,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            collate_fn=collate_return,
        )


class TableClsRegDataModule(ClsRegDataModule):
    """ClsRegDataModule for image files, such as PNGs, whose labels are in a CSV table."""

    dataset_class = TableClsRegDataset
    test_dataset_class = TableClsRegTestDataset

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        label_table: str,
        label_column: str,
        label_cutoff: float,
        file_column: str = "file_path",
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
        test_samples: Optional[list] = [],
        predict_samples: Optional[list] = [],
        use_random_datasampler: Optional[bool] = True,
    ):
        """
        Args:
            label_table: CSV with a header and a row per file.
            label_column: Column of label_table with the labels.
            label_cutoff: Labels below it become 1 and the rest 0.
            file_column: Column of label_table with the file paths, as in the splits.
            The other arguments are those of ClsRegDataModule.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=train_split,
            val_split=val_split,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            predict_transforms=predict_transforms,
            test_samples=test_samples,
            predict_samples=predict_samples,
            use_random_datasampler=use_random_datasampler,
        )
        table = polars.read_csv(label_table, columns=[file_column, label_column])
        table = table.select(file_path=polars.col(file_column), label=polars.col(label_column))
        # Sorted so the datasets find a file's row by binary search
        self.table = table.sort("file_path")
        self.label_cutoff = label_cutoff

    def setup_fit(self):
        self.train_dataset = self.dataset_class(
            self.train_split,
            table=self.table,
            label_cutoff=self.label_cutoff,
            transforms=self.train_transforms,
        )
        self.val_dataset = self.dataset_class(
            self.val_split,
            table=self.table,
            label_cutoff=self.label_cutoff,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = self.test_dataset_class(
            self.test_samples,
            table=self.table,
            label_cutoff=self.label_cutoff,
            transforms=self.test_transforms,
        )
