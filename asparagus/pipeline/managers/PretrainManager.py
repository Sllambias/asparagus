import lightning as L
import torch
import wandb
import logging
import yucca
from batchgenerators.utilities.file_and_folder_operations import join
from typing import Literal, Union, Optional
from asparagus.modules.lightning_modules import SelfSupervisedModel
from asparagus.modules.datasets import PretrainDataset
from asparagus.modules.datamodules import PretrainDataModule


class PretrainManager:
    def __init__(
        self,
        task: str,
        accelerator: str = "auto",
        data_module: pl.LightningDataModule = PretrainDataModule,
        max_epochs: int = 5,
        num_workers: Optional[int] = None,
        precision: str = "bf16-mixed",
        **kwargs,
    ):
        self.task = task
        self.accelerator = accelerator
        self.data_module_class = data_module
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.precision = precision
        self.kwargs = kwargs

        # Initialize other parameters
        self.train_batches_per_step = 250
        self.val_batches_per_step = 50
        self.enable_logging = True
        self.enable_progress_bar = True
        self.profile = False
        self.wandb_log_model = True

    def initialize(
        self,
    ):
       
        callbacks, loggers, profilers = (None, None, None)

        self.model_module = self.lightning_module(
        )

        self.data_module = self.data_module_class(
        )


        self.trainer = L.Trainer(
            accelerator=self.accelerator,
            callbacks=callbacks,
            default_root_dir=path_config.save_dir,
            limit_train_batches=self.train_batches_per_step,
            limit_val_batches=self.val_batches_per_step,
            log_every_n_steps=min(self.train_batches_per_step, 50),
            logger=loggers,
            precision=self.precision,
            profiler=profiler,
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.max_epochs,
            devices=1,
            **self.kwargs,
        )

    def run_training(self):
        self.initialize(stage="fit")
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
            ckpt_path="last",
        )
        self.finish()

if __name__ == "__main__":
