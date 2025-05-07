from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig

GBrainPreprocessingConfig = PreprocessingConfig(
    normalization_operation=["no_norm"], target_spacing=None, target_orientation="RAS"
)
