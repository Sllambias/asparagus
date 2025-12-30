import hydra
import lightning as pl
import os
import random
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import FinetuneSearchpathPlugin
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.experiment_setup import (
    prepare_standard_experiment,
)
from asparagus.pipeline.auto_configuration.logging import logging
from dotenv import load_dotenv
from gardening_tools.modules.networks.components.weight_init import set_params_to_zero
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("version", lambda: generate_unused_run_id(), use_cache=True)
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(FinetuneSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_cls",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project="Finetune",
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
    )

    best_ckpt_callback = ModelCheckpoint(
        dirpath=path_store.ckpt_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=path_store.ckpt_save_dir,
        every_n_epochs=cfg.model.ckpt_every_n_epoch,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )

    progressbar_callback = TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    profilers = None

    cpu_tr_transforms = instantiate(
        cfg.transforms._cpu_tr_transforms,
        target_size=cfg.training.target_size,
    )
    cpu_val_transforms = instantiate(
        cfg.transforms._cpu_val_transforms,
        target_size=cfg.training.target_size,
    )
    gpu_tr_transforms = instantiate(cfg.transforms._gpu_tr_transforms, ndim=len(cfg.training.target_size))

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=file_store.test,
        test_transforms=CPU_clsreg_val_test_transforms_crop(target_size=cfg.training.target_size),
    )

    model = instantiate(
        cfg.model._cls_net,
        input_channels=file_store.dataset_json["metadata"]["n_modalities"],
        output_channels=file_store.dataset_json["metadata"]["n_classes"],
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        warmup_epochs=cfg.training.warmup_epochs,
        decoder_warmup_epochs=cfg.training.decoder_warmup_epochs,
        train_transforms=gpu_tr_transforms,
        val_transforms=None,
        weights=path_store.ckpt_path,
        log_image_every_n_epochs=cfg.logger.log_images_every_n_epoch,
        optimizer=cfg.model.finetune_optim,
        learning_rate=cfg.model.finetune_lr,
        repeat_stem_weights=cfg.training.repeat_stem_weights,
        test_output_path=os.path.join(
            path_store.run_dir,
            "predictions",
            cfg.test_task + "__" + cfg.data.test_split + "__" + "best.json",
        ),
    )

    trainer = instantiate(
        cfg.lightning._trainer,
        callbacks=[
            last_ckpt_callback,
            best_ckpt_callback,
            progressbar_callback,
            lr_monitor_callback,
        ],
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        profiler=profilers,
        default_root_dir=path_store.run_dir,
        max_epochs=cfg.training.epochs,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        use_distributed_sampler=False,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
    )

    model_module.model.apply(set_params_to_zero)

    trainer.test(
        model=model_module,
        datamodule=data_module,
        ckpt_path=best_ckpt_callback.best_model_path,
    )


if __name__ == "__main__":
    main()
