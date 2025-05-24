from dataclasses import dataclass


@dataclass
class PathingConfig:
    output_dir: str
    ckpt_path: str
    dataset_json_path: str
