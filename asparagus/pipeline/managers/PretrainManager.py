# %%
import lightning as pl
import torch
import wandb
import logging
import yucca
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from typing import Literal, Union, Optional
from asparagus.modules.lightning_modules import SelfSupervisedModel
from asparagus.modules.datasets import PretrainDataset
from asparagus.modules.datamodules import PretrainDataModule
from asparagus.modules.networks.nets.unet import unet_b_lw_dec
from asparagus.paths import get_data_path

# %%


class PretrainManager:
    def __init__(
        self,
        task: str,
        accelerator: str = "cpu",
        data_module: pl.LightningDataModule = PretrainDataModule,
        max_epochs: int = 5,
        num_workers: Optional[int] = 1,
        lightning_module: pl.LightningModule = SelfSupervisedModel,
        precision: str = "bf16-mixed",
        splits: str = "split_80_20.json",
        **kwargs,
    ):
        self.task = task
        self.accelerator = accelerator
        self.data_module = data_module
        self.lightning_module = lightning_module
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.precision = precision
        self.kwargs = kwargs

        # Initialize other parameters
        self.limit_train_batches = 5
        self.limit_val_batches = 5
        self.enable_logging = True
        self.enable_progress_bar = True
        self.profile = False
        self.wandb_log_model = True

        # Hardcoded
        self.batch_size = 2
        self.model = unet_b_lw_dec
        self.learning_rate = 1e-4
        self.patch_size = (128, 128, 128)

        self.data_path = join(get_data_path(), self.task)
        self.splits = load_json(join(self.data_path, splits))

    def initialize(
        self,
    ):

        self.steps_per_epoch = len(self.splits["train"]) // self.batch_size

        callbacks, loggers, profilers = (None, None, None)

        self.data_module = self.data_module(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            patch_size=self.patch_size,
            train_split=self.splits["train"],
            val_split=self.splits["val"],
        )

        self.model_module = self.lightning_module(
            model=self.model,
            epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=callbacks,
            # default_root_dir=path_config.save_dir,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            log_every_n_steps=100,
            logger=loggers,
            precision=self.precision,
            profiler=profilers,
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.max_epochs,
            devices=1,
            **self.kwargs,
        )

    def run_training(self):
        self.initialize()
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
            ckpt_path="last",
        )


if __name__ == "__main__":
    man = PretrainManager(task="Task998_LauritSyn")
    man.run_training()


# %%
