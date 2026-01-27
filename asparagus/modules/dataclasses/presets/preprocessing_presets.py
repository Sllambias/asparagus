from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from typing import List, Optional

GBrainPreprocessingConfig = PreprocessingConfig(
    normalization_operation=["no_norm"], target_spacing=None, target_orientation="RAS"
)
GBrainPreprocessingConfig3Mods = PreprocessingConfig(
    normalization_operation=["no_norm", "no_norm", "no_norm"], target_spacing=None, target_orientation="RAS"
)

GBrainPreprocessingConfig4Mods = PreprocessingConfig(
    normalization_operation=["no_norm", "no_norm", "no_norm", "no_norm"], target_spacing=None, target_orientation="RAS"
)

GBrainPreprocessingConfig5Mods = PreprocessingConfig(
    normalization_operation=["no_norm", "no_norm", "no_norm", "no_norm", "no_norm"],
    target_spacing=None,
    target_orientation="RAS",
)

ClsPreprocessingConfig = PreprocessingConfig(
    normalization_operation=["no_norm"], target_spacing=None, target_orientation="RAS", target_size=[128.0, 128.0, 128.0]
)


def get_preprocessing_config(
    spacing: Optional[List[float]] = None, modalities=1, norm_op: str = "no_norm"
) -> PreprocessingConfig:
    return PreprocessingConfig(
        normalization_operation=[norm_op] * modalities, target_spacing=spacing, target_orientation="RAS"
    )
