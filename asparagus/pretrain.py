import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
import lightning as pl
import random
from asparagus.pipeline.auto_configuration import logging
from hydra.core.hydra_config import HydraConfig
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_standard_experiment, prepare_online_segmentation
from asparagus.paths import get_config_path
from asparagus.modules.callbacks.ssl_training import OnlineSegmentationPlugin
from asparagus.modules.data_modules.training import TrainDataModule
from asparagus.modules.networks.nets.unet import unet_b

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))


@hydra.main(
    config_path=get_config_path(),
    config_name="pretrain",
    version_base="1.2",
)
def train(cfg: DictConfig) -> None:
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    seg_plugin = prepare_online_segmentation(cfg)
    steps_per_epoch = len(file_store.splits["train"]) // cfg.training.batch_size
    pl.seed_everything(seed=cfg.experiment.seed, workers=True)

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        save_dir=path_store.output_dir,
        steps_per_epoch=steps_per_epoch,
        version=version_store.version,
        version_dir=version_store.version_dir,
        wandb_experiment=HydraConfig.get().job.config_name,
    )

    seg_data_module = TrainDataModule(
        train_split=seg_plugin.splits["train"],
        val_split=seg_plugin.splits["val"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        patch_size=cfg.training.patch_size,
        composed_train_transforms=None,
        composed_val_transforms=None,
    )

    callbacks = [
        OnlineSegmentationPlugin(
            data_module=seg_data_module,
            epochs=3,
            every_n_epochs=1,
            train_n_last_params=6,
            model_class=unet_b,
            dimensions="2D",
        )
    ]
    profilers = None

    model = instantiate(
        cfg._internal_.net,
        input_channels=cfg.model.input_channels,
        output_channels=cfg.model.output_channels,
        dimensions=cfg.model.dimensions,
    )

    model_module = instantiate(
        cfg._internal_.lightning_module,
        model=model,
        steps_per_epoch=steps_per_epoch,
    )

    data_module = instantiate(
        cfg._internal_.data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
    )

    trainer = instantiate(
        cfg._internal_.trainer,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=loggers,
        profiler=profilers,
        default_root_dir=path_store.output_dir,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )


if __name__ == "__main__":
    train()
