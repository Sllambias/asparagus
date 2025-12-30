import hydra
import lightning as pl
import random
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import PretrainSearchpathPlugin
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_ssl_plugins, prepare_standard_experiment
from asparagus.pipeline.auto_configuration.logging import logging
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("version", lambda: generate_unused_run_id(), use_cache=True)
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(PretrainSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    plugins = prepare_ssl_plugins(cfg)

    assert cfg.task is not None, "Config file is not set up correctly."

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project="Pretrain",
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
    )

    callbacks = [
        TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps),
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            every_n_epochs=cfg.model.ckpt_every_n_epoch,
            save_top_k=1,
            filename="last",
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
    ] + plugins

    if cfg.profiler.enabled:
        callbacks.append(instantiate(cfg.profiler._callback))

    cpu_tr_transforms = instantiate(
        cfg.transforms._cpu_tr_transforms,
        patch_size=cfg.training.patch_size,
    )
    cpu_val_transforms = instantiate(
        cfg.transforms._cpu_val_transforms,
        patch_size=cfg.training.patch_size,
    )
    gpu_tr_transforms = instantiate(
        cfg.transforms._gpu_tr_transforms,
        cfg.transforms.masking,
        ndim=len(cfg.training.patch_size),
        mask_ratio=cfg.training.mask_ratio,
    )
    gpu_val_transforms = instantiate(
        cfg.transforms._gpu_val_transforms,
        cfg.transforms.masking,
        mask_ratio=cfg.training.mask_ratio,
    )

    model = instantiate(
        cfg.model._pretrain_net,
        input_channels=1,
        output_channels=1,
    )

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        learning_rate=cfg.model.pretrain_lr,
        warmup_epochs=int(cfg.training.epochs * cfg.training.warmup_ratio),
        train_transforms=gpu_tr_transforms,
        val_transforms=gpu_val_transforms,
        rec_loss_masked_only=cfg.training.rec_loss_masked_only,
        optimizer=cfg.model.pretrain_optim,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        log_images_every_n_epoch=cfg.logger.log_images_every_n_epoch,
    )

    trainer = instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        default_root_dir=path_store.run_dir,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        max_epochs=cfg.training.epochs,
        limit_train_batches=cfg.training.steps_per_epoch,
        limit_val_batches=cfg.training.val_steps_per_epoch,
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
