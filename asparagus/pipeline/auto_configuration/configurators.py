from yucca.pipeline.configuration.configure_paths import detect_version
from asparagus.modules.dataclasses.file_management_and_version_control import (
    FileManagementAndVersionControlConfig,
)
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)


def file_management_version_control_configurator(
    save_dir: str,
    train_data_dir: str,
    experiment: str = "default",
    version: int = 0,
    continue_from_most_recent: bool = True,
) -> FileManagementAndVersionControlConfig:
    """
    Configurator for file management and version control.
    """
    # Create the save directory if it doesn't exist
    ensure_dir_exists(save_dir)

    # Detect the version of the data
    version = detect_version(save_dir, continue_from_most_recent)

    # Create a simple path configuration
    cfg = FileManagementAndVersionControlConfig(
        save_dir=save_dir,
        experiment=experiment,
        version=version,
        train_data_dir=train_data_dir,
        continue_from_most_recent=continue_from_most_recent,
    )

    return cfg
