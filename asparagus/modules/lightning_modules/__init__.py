from .clsreg_module import ClassificationModule, RegressionModule
from .dinov2 import DINOv2Module
from .linear_probe_module import LinearProbeModule
from .segmentation_module import SegmentationModule
from .self_supervised import SelfSupervisedModule

__all__ = [
    "SegmentationModule",
    "ClassificationModule",
    "RegressionModule",
    "SelfSupervisedModule",
    "DINOv2Module",
    "LinearProbeModule",
]
