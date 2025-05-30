import lightning as pl
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
from lightning.pytorch.callbacks import TQDMProgressBar
from asparagus.pipeline.auto_configuration import logging
from hydra.core.hydra_config import HydraConfig
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_standard_experiment, prepare_ssl_plugins
from asparagus.paths import get_config_path
from asparagus.modules.callbacks.ssl_training import OnlineSegmentationPlugin
from asparagus.modules.data_modules.training import TrainDataModule
from asparagus.modules.networks.nets.unet import unet_b
from asparagus.asparagus.modules.hydra.plugins.searchpath_plugins import ExampleSearchPathPlugin
from hydra.core.plugins import Plugins

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
Plugins.instance().register(ExampleSearchPathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="main_pretrain",
    version_base="1.2",
)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    plugins = prepare_ssl_plugins(cfg)
    steps_per_epoch = len(file_store.splits["train"]) // cfg.training.batch_size
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        save_dir=path_store.output_dir,
        steps_per_epoch=steps_per_epoch,
        version=version_store.version,
        version_dir=version_store.version_dir,
        wandb_experiment=HydraConfig.get().job.config_name,
    )

    callbacks = [TQDMProgressBar(refresh_rate=50)] + plugins
    profilers = None

    model = instantiate(
        cfg._model,
        input_channels=1,
        output_channels=1,
    )

    model_module = instantiate(
        cfg._lightning_module,
        model=model,
        steps_per_epoch=steps_per_epoch,
    )

    data_module = instantiate(
        cfg._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
    )

    trainer = instantiate(
        cfg._trainer,
        callbacks=callbacks,
        log_every_n_steps=250,
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
