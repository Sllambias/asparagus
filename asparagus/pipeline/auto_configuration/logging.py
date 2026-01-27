import math
from asparagus.functional.decorators import depends_on_mlflow
from asparagus.modules.callbacks.loggers import BaseLogger
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from typing import Optional, Union


@depends_on_mlflow()
class SafeMLFlowLogger(MLFlowLogger):
    def log_metrics(self, metrics, step=None):
        safe_metrics = {
            k.replace("/", "_"): (-99999 if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()
        }
        super().log_metrics(safe_metrics, step)


def logging(
    ckpt_wandb_id: Union[str, None],
    ckpt_mlflow_id: Union[str, None],
    log_file_name: str,
    run_dir: str,
    version: Union[int, str],
    wandb_experiment: str,
    wandb_run_description: str = None,
    wandb_project: str = "Asparagus",
    wandb_entity: Optional[str] = None,
    wandb_log_model: Union[bool, str] = False,
    wandb_logging: bool = True,
    wandb_config: dict = None,
    mlflow_logging: bool = False,
):
    """
    Configure and return loggers for training.

    Args:
        ckpt_wandb_id: ID for checkpoint (used by wandb)
        run_dir: Directory to save logs
        version: Version identifier
        wandb_experiment: Experiment name
        logger_type: Type of logger to use ('wandb' or 'mlflow')

    Returns:
        list: Configured loggers
    """
    loggers = [BaseLogger(save_dir=run_dir, file_name=log_file_name)]

    if wandb_logging:
        loggers.append(
            WandbLogger(
                name=f"{wandb_experiment}_{version}",
                notes=wandb_run_description,
                save_dir=run_dir,
                project=wandb_project,
                group=wandb_experiment,
                log_model=wandb_log_model,
                version=ckpt_wandb_id if ckpt_wandb_id else None,
                resume="allow" if ckpt_wandb_id else None,
                entity=wandb_entity,
                config=wandb_config,
            )
        )

    if mlflow_logging:
        loggers.append(
            SafeMLFlowLogger(
                experiment_name=wandb_experiment,
                tracking_uri=f"file:{run_dir}/mlruns",
                run_id=ckpt_mlflow_id if ckpt_mlflow_id else None,
            )
        )

    return loggers
