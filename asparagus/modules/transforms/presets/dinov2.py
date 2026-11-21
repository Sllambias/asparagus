from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from typing import Any, List, Optional, Tuple, Union


def dinov2(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    **kwargs: Any,
) -> DINOv2Augmentation:
    """The official DINOv2 augmentation. The other arguments go to DINOv2Augmentation."""
    return DINOv2Augmentation(global_view_size=global_view_size, local_view_size=local_view_size, **kwargs)


def ultradino(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    **kwargs: Any,
) -> DINOv2Augmentation:
    """UltraDino's augmentation (DataAugmentationDINUS in Dinus): the official one with vertical flips, no hue jitter,
    no RandomGrayscale and the FUS13M mean and std. The other arguments go to DINOv2Augmentation."""
    return DINOv2Augmentation(
        global_view_size=global_view_size,
        local_view_size=local_view_size,
        vertical_flip=True,
        hue=0.0,
        grayscale_prob=0.0,
        normalize_mean=[0.1842924807],
        normalize_std=[0.2187705424],
        **kwargs,
    )
