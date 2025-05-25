import lightning as pl
from torchvision.transforms import Compose
import logging
from typing import Literal, Optional, Tuple
from torch.utils.data import DataLoader
from asparagus.modules.datasets.TrainDataset import TrainDataset


class TrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        patch_size: Tuple[int, int, int],
        train_split: list,
        val_split: list,
        composed_train_transforms: Optional[Compose] = None,
        composed_val_transforms: Optional[Compose] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.train_split = train_split
        self.val_split = val_split

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):

        self.train_dataset = TrainDataset(
            self.train_split,
            composed_transforms=self.composed_train_transforms,
            patch_size=self.patch_size,
        )

        self.val_dataset = TrainDataset(
            self.val_split,
            composed_transforms=self.composed_val_transforms,
            patch_size=self.patch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    from batchgenerators.utilities.file_and_folder_operations import (
        load_json,
    )

    dataset_json = load_json(
        "/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task997_LauritSynSeg/dataset.json"
    )
    splits = load_json(
        "/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task997_LauritSynSeg/split_80_20.json"
    )[0]
    train_split = splits["train"]
    val_split = splits["val"]
    data_module = TrainDataModule(
        train_split=train_split,
        val_split=val_split,
        patch_size=(32, 32),
        batch_size=2,
        num_workers=6,
    )
    data_module.setup("fit")
    data_module_iterator = iter(data_module.train_dataloader())
    x = next(data_module_iterator)
    print(type(x))
    print(next(iter(data_module.train_dataset))["image"].shape)
