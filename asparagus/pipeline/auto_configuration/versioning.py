from asparagus.functional.versioning import detect_version, detect_wandb_id, detect_used_ids, detect_id
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
    run_id = detect_used_ids(save_dir=save_dir, continue_from_most_recent=continue_from_most_recent)
    run_id_dir = join(save_dir, f"run_id={run_id}")
    wandb_id = detect_wandb_id(version_dir=run_id_dir, continue_from_most_recent=continue_from_most_recent)
    # Create a simple path configuration
    cfg = VersioningConfig(
        version=run_id,
        version_dir=run_id_dir,
        wandb_id=wandb_id,
    )

    return cfg


def pathing(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.experiment.pretrained_run_id is not None:
        model_folder = detect_id(cfg.experiment.pretrained_run_id)
        pretrained_ckpt = join(model_folder, "checkpoints", cfg.experiment.pretrained_checkpoint_name)
    else:
        model_folder, pretrained_ckpt = "", ""
    dataset_json_path = cfg.data.data_path + "/dataset.json"
    pathingcfg = PathingConfig(
        output_dir=output_dir, ckpt_parent_folder=model_folder, ckpt_path=pretrained_ckpt, dataset_json_path=dataset_json_path
    )
    return pathingcfg
