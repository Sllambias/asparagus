from .clsreg_module import ClassificationModule, RegressionModule
from .linear_probe_module import LinearProbeModule
from .segmentation_module import SegmentationModule
from .self_supervised import SelfSupervisedModule
from importlib import import_module

__all__ = [
    "SegmentationModule",
    "ClassificationModule",
    "RegressionModule",
    "SelfSupervisedModule",
    "DINOv2Module",
    "LinearProbeModule",
]


def __getattr__(name):
    """Lazily import optional-dependency-backed Lightning modules."""
    if name == "DINOv2Module":
        module = import_module(".dinov2", __name__)
        value = module.DINOv2Module
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
