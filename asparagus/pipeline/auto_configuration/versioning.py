import os
from asparagus.functional.versioning import detect_id, detect_mlflow_id, detect_wandb_id
from asparagus.modules.dataclasses import PathingConfig, VersioningConfig
from asparagus.pipeline.auto_configuration.checkpoint import resolve_checkpoint_path
from hydra.core.hydra_config import HydraConfig


def versioning(cfg) -> VersioningConfig:
    """
    Configurator for file management and version control.
    """
    run_dir = HydraConfig.get().runtime.output_dir

    run_id = cfg.run_id
    wandb_id = detect_wandb_id(run_dir=run_dir)
    mlflow_id = detect_mlflow_id(run_dir=run_dir)

    return VersioningConfig(version=run_id, wandb_id=wandb_id, mlflow_id=mlflow_id)


def pathing(cfg, train=True):
    run_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(cfg)
    ckpt_parent_folder = detect_id(cfg.checkpoint_run_id) if cfg.checkpoint_run_id else None

    if train:
        dataset_json_path = cfg.data.data_path + "/dataset.json"
    else:
        dataset_json_path = cfg.data.test_data_path + "/dataset.json"

    return PathingConfig(
        run_dir=run_dir,
        ckpt_save_dir=os.path.join(run_dir, "checkpoints"),
        ckpt_parent_folder=ckpt_parent_folder,
        ckpt_path=ckpt_path,
        dataset_json_path=dataset_json_path,
    )
