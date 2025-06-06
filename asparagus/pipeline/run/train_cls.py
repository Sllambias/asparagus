import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
import lightning as pl
import random
from asparagus.pipeline.auto_configuration import versioning, logging
from hydra.core.hydra_config import HydraConfig
from asparagus.functional.utils import add_run_to_pretrained_derivative_list
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_standard_experiment
from asparagus.paths import get_config_path
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))


@hydra.main(
    config_path=get_config_path(),
    config_name="main_train",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    file_store, path_store, version_store = prepare_standard_experiment(cfg)

    steps_per_epoch = len(file_store.splits["train"]) // cfg.training.batch_size
    pl.seed_everything(seed=cfg.training.training.seed, workers=True)

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        save_dir=path_store.output_dir,
        version=version_store.version,
        version_dir=version_store.version_dir,
        wandb_experiment=HydraConfig.get().job.config_name,
    )

    callbacks = [TQDMProgressBar(refresh_rate=100)]
    profilers = None

    model = instantiate(
        cfg.model._train_cls_net,
        input_channels=file_store.dataset_json["metadata"]["n_modalities"],
        output_channels=file_store.dataset_json["metadata"]["n_classes"],
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        steps_per_epoch=steps_per_epoch,
    )

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
    )

    trainer = instantiate(
        cfg.lightning._trainer,
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
    main()
