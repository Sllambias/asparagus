from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple


@dataclass
class TrainingFiles:
    dataset_json: None
    splits: None


@dataclass
class SegmentationPlugin:
    dataset_json: None
    splits: None
    num_classes: int
    num_modalities: int
