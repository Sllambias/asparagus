from asparagus.functional.versioning import detect_version, detect_wandb_id
from asparagus.modules.dataclasses.versioning import (
    VersioningConfig,
)
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
    join,
)


def versioning(
    save_dir: str,
    continue_from_most_recent: bool = True,
) -> VersioningConfig:
    """
    Configurator for file management and version control.
    """
    # Create the save directory if it doesn't exist
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
