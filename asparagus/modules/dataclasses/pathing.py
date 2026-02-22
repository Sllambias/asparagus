from dataclasses import dataclass
from typing import Optional


@dataclass
class PathingConfig:
    run_dir: str
    ckpt_save_dir: str
    ckpt_path: Optional[str]
    ckpt_parent_folder: Optional[str]
    dataset_json_path: str
