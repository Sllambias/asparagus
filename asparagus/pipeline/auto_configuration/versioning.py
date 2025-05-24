from asparagus.functional.versioning import detect_version, detect_wandb_id
from asparagus.modules.dataclasses import VersioningConfig, PathingConfig
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists, join, load_json
from hydra.core.hydra_config import HydraConfig


def versioning(cfg) -> VersioningConfig:
    """
    Configurator for file management and version control.
    """
    # Create the save directory if it doesn't exist
    save_dir = HydraConfig.get().runtime.output_dir
    continue_from_most_recent = cfg.experiment.resume_training
    ensure_dir_exists(save_dir)

    # Detect the version of the data
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = join(save_dir, f"version_{version}")
    wandb_id = detect_wandb_id(version_dir=version_dir, continue_from_most_recent=continue_from_most_recent)
    # Create a simple path configuration
    cfg = VersioningConfig(
        version=version,
        version_dir=version_dir,
        wandb_id=wandb_id,
    )

    return cfg


def pathing(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    pretrained_ckpt = cfg.experiment.checkpoint
    dataset_json_path = cfg.data.data_path + "/dataset.json"
    pathingcfg = PathingConfig(output_dir=output_dir, ckpt_path=pretrained_ckpt, dataset_json_path=dataset_json_path)
    return pathingcfg
