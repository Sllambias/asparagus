from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from typing import Any, List, Optional, Tuple, Union


def dinov2(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    num_local_views: int = 8,
    **kwargs: Any,
) -> DINOv2Augmentation:
    """DINOv2 multi-crop augmentation with the official 8 local crops. The other arguments go to DINOv2Augmentation."""
    return DINOv2Augmentation(
        global_view_size=global_view_size,
        local_view_size=local_view_size,
        num_local_views=num_local_views,
        **kwargs,
    )
