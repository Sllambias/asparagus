from dataclasses import dataclass


@dataclass
class FileManagementAndVersionControlConfig:
    save_dir: str
    train_data_dir: str
    experiment: str = "default"
    version: int = 0
    continue_from_most_recent: bool = True
