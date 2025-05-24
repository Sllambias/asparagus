import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
import lightning as pl
import random
from asparagus.pipeline.auto_configuration import versioning, logging
from hydra.core.hydra_config import HydraConfig

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))


@hydra.main(
    config_path="/Users/zcr545/Desktop/Projects/repos/asparagus_data/configs",
    config_name="pretrain",
    version_base="1.2",
)
def train(cfg: DictConfig) -> None:
    splits = load_json(cfg._internal_.splits_path)
    steps_per_epoch = len(splits["train"]) // cfg.training.batch_size
    output_dir = HydraConfig.get().runtime.output_dir

    version_cfg = versioning(save_dir=output_dir, continue_from_most_recent=cfg.experiment.resume_training)
    loggers = logging(
        ckpt_wandb_id=version_cfg.wandb_id,
        save_dir=output_dir,
        steps_per_epoch=steps_per_epoch,
        version=version_cfg.version,
        version_dir=version_cfg.version_dir,
        wandb_experiment=HydraConfig.get().job.config_name,
    )
    callbacks, profilers = (None, None)

    pl.seed_everything(seed=cfg.experiment.seed, workers=True)

    model = instantiate(
        cfg._internal_.net,
        input_channels=cfg.,
        output_channels=1,
        dimensions=cfg.model.dimensions,
    )

    model_module = instantiate(
        cfg._internal_.lightning_module,
        model=model,
        steps_per_epoch=steps_per_epoch,
    )

    data_module = instantiate(
        cfg._internal_.data_module,
        train_split=splits["train"],
        val_split=splits["val"],
    )

    trainer = instantiate(
        cfg._internal_.trainer,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=loggers,
        profiler=profilers,
        default_root_dir=output_dir,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )


if __name__ == "__main__":
    train()
