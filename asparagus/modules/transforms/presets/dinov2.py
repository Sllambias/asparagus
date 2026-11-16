from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from typing import List, Optional, Tuple, Union


def dinov2(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    global_view_scale: Tuple[float, float] = (0.32, 1.0),
    local_view_scale: Tuple[float, float] = (0.05, 0.32),
    num_local_views: int = 8,
) -> DINOv2Augmentation:
    """DINOv2 multi-crop augmentation with the official crop scales and number of local crops."""
    return DINOv2Augmentation(
        global_view_scale=list(global_view_scale),
        global_view_size=global_view_size,
        local_view_scale=list(local_view_scale),
        local_view_size=local_view_size,
        num_local_views=num_local_views,
    )
