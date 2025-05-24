from dataclasses import dataclass


@dataclass
class VersioningConfig:
    version: int
    version_dir: str
    wandb_id: str
