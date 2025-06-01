from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig

GBrainPreprocessingConfig = PreprocessingConfig(
    normalization_operation=["no_norm"], target_spacing=None, target_orientation="RAS"
)

ClsPreprocessingConfig = PreprocessingConfig(
    normalization_operation=["no_norm"], target_spacing=None, target_orientation="RAS", target_size=[128.0, 128.0, 128.0]
)
