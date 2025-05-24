import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
import lightning as pl

load_dotenv()


@hydra.main(
    config_path="/Users/zcr545/Desktop/Projects/repos/asparagus/asparagus/pipeline/configs",
    config_name="pretrain",
    version_base="1.1",
)
def train(cfg: DictConfig) -> None:
    splits = load_json(cfg._internal_.splits_path)
    steps_per_epoch = len(splits["train"]) // cfg.training.batch_size
    callbacks, loggers, profilers = (None, None, None)

    pl.seed_everything(seed=cfg.experiment.seed, workers=True)

    model = instantiate(
        cfg._internal_.net,
        input_channels=1,
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
        log_every_n_steps=100,
        logger=loggers,
        profiler=profilers,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )


if __name__ == "__main__":
    train()
