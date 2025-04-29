from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple


@dataclass
class DataConfig:
    input_channels: int
    num_classes: int
    train_dataset_size: int
    val_dataset_size: int

@dataclass
class ModelConfig:
    model_name:str
    model_dimensions: str = "3D"

@dataclass
class PretrainingConfig:
    task_type:str = "self-supervised"
    patch_size: Tuple(int, int, Optional[int])
    mask_patch_size: int
    mask_ratio: float
    exclude_nonmasked_tokens_from_rec_loss: bool
    batch_size: int
    epochs: int
    warmup_epochs: int
    learning_rate: float 
    optimizer: str # should this just be an optimizer instance instead of a str?
    effective_batch_size: int
    precision: str
    augmentation_preset: str, # this should be the dict/set of parameters for the augmentation preset
    fast_dev_run: bool
    limit_val_batches: int = None
    limit_train_batches: int = None
    overfit_batches: int = 0
    check_val_every_n_epoch: int = None
    accumulate_grad_batches: int = 1

@dataclass
class FinetuningConfig:
    pass