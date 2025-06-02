from typing import Union, Optional

# from yucca.modules.callbacks.loggers import YuccaLogger
from asparagus.modules.callbacks.loggers import BaseLogger
from lightning.pytorch.loggers import WandbLogger


def logging(
    ckpt_wandb_id: Union[str, None],
    save_dir: str,
    version_dir: Union[str, None],
    version: Union[int, str],
    wandb_experiment: str,
    wandb_run_description: str = None,
    wandb_project: str = "Asparagus",
    wandb_entity: Optional[str] = None,
    wandb_log_model: Union[bool, str] = False,
):

    loggers = [
        BaseLogger(
            save_dir=save_dir,
            name=None,
            version=f"run_id={version}",
        )
    ]
    loggers.append(
        WandbLogger(
            name=f"{wandb_experiment}_{version}",
            notes=wandb_run_description,
            save_dir=version_dir,
            project=wandb_project,
            group=wandb_experiment,
            log_model=wandb_log_model,
            version=ckpt_wandb_id if ckpt_wandb_id else None,
            resume="allow" if ckpt_wandb_id else None,
            entity=wandb_entity,
        )
    )

    return loggers
