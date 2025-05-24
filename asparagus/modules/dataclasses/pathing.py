from dataclasses import dataclass


@dataclass
class PathingConfig:
    output_dir: str
    ckpt_path: str
    ckpt_parent_folder: str
    dataset_json_path: str
