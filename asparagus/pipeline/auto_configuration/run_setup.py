import os
import random
from asparagus.functional.hydra import fast_instantiate
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.pipeline.auto_configuration.logging import logging
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from omegaconf import OmegaConf


def register_resolvers():
    """Register the OmegaConf resolvers the configs use, skipping any that are already registered."""
    if not OmegaConf.has_resolver("random"):
        OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
    if not OmegaConf.has_resolver("version"):
        OmegaConf.register_new_resolver(
            "version",
            lambda resume_training, run_dir: generate_unused_run_id(resume_training=resume_training, run_dir=run_dir),
            use_cache=True,
        )
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def build_loggers(cfg, path_store, version_store, **kwargs):
    """Loggers with the arguments every training script passes. kwargs adds the ones only some scripts pass."""
    return logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project=cfg.logger.wandb_project,
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
        **kwargs,
    )


def build_checkpoint_callbacks(cfg, path_store):
    """Callbacks that keep the checkpoint with the lowest val/loss and the last checkpoint."""
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
    return best_ckpt_callback, last_ckpt_callback


def build_monitoring_callbacks(cfg):
    progressbar_callback = TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    return [progressbar_callback, lr_monitor_callback]


def build_trainer(cfg, path_store, loggers, callbacks, **kwargs):
    """The Lightning trainer with the arguments every training script passes. kwargs adds the rest."""
    return fast_instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        profiler=None,
        default_root_dir=path_store.run_dir,
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        **kwargs,
    )


def test_output_path(cfg, path_store, checkpoint_name):
    file_name = cfg.test_task + "__" + cfg.data.test_split + "__" + checkpoint_name + ".json"
    return os.path.join(path_store.run_dir, "predictions", file_name)


def checkpoint_to_test(cfg, best_ckpt_callback, last_ckpt_callback):
    """The checkpoint path picked by cfg.test_checkpoint (best or last)."""
    if cfg.test_checkpoint == "best":
        test_ckpt_callback = best_ckpt_callback
    else:
        test_ckpt_callback = last_ckpt_callback
    assert test_ckpt_callback.best_model_path, f"No '{cfg.test_checkpoint}' checkpoint was saved to test from."
    return test_ckpt_callback.best_model_path
