from dataclasses import dataclass


@dataclass
class VersioningConfig:
    version: int
    wandb_id: str
    mlflow_id: int
