import os
from asparagus.functional.versioning import detect_id, detect_mlflow_id, detect_wandb_id
from asparagus.modules.dataclasses import PathingConfig, VersioningConfig
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

    if cfg.checkpoint_run_id is not None and cfg.checkpoint_run_id != "":
        model_folder = detect_id(cfg.checkpoint_run_id)
        pretrained_ckpt = os.path.join(model_folder, "checkpoints", cfg.load_checkpoint_name)
        assert cfg.checkpoint_path is None, "You cannot provide both a checkpoint path and a checkpoint run id"
    elif cfg.checkpoint_path is not None and cfg.checkpoint_path != "":
        model_folder = None
        pretrained_ckpt = cfg.checkpoint_path
    else:
        model_folder, pretrained_ckpt = None, None

    if train:
        dataset_json_path = cfg.data.data_path + "/dataset.json"
    else:
        dataset_json_path = cfg.data.test_data_path + "/dataset.json"
    pathingcfg = PathingConfig(
        run_dir=run_dir,
        ckpt_save_dir=os.path.join(run_dir, "checkpoints"),
        ckpt_parent_folder=model_folder,
        ckpt_path=pretrained_ckpt,
        dataset_json_path=dataset_json_path,
    )
    return pathingcfg
