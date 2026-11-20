from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from typing import List, Optional, Sequence, Tuple, Union


def dinov2(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    global_view_scale: Tuple[float, float] = (0.32, 1.0),
    local_view_scale: Tuple[float, float] = (0.05, 0.32),
    num_local_views: int = 8,
    resize_mode: Optional[str] = None,
    flip_axis: Optional[int] = None,
    affine_prob: float = 0.5,
    histogram_shift_prob: float = 0.5,
    color_jitter_prob: float = 0.0,
    brightness: float = 0.4,
    contrast: float = 0.4,
    blur_probs: Sequence[float] = (1.0, 0.1, 0.5),
    blur_sigma: Optional[Tuple[float, float]] = None,
    solarize_prob: float = 0.2,
    solarize_threshold: float = 0.5,
    value_range: Optional[Tuple[float, float]] = None,
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
) -> DINOv2Augmentation:
    """DINOv2 multi-crop augmentation with the official crop scales and number of local crops. See DINOv2Augmentation."""
    return DINOv2Augmentation(
        global_view_scale=list(global_view_scale),
        global_view_size=global_view_size,
        local_view_scale=list(local_view_scale),
        local_view_size=local_view_size,
        num_local_views=num_local_views,
        resize_mode=resize_mode,
        flip_axis=flip_axis,
        affine_prob=affine_prob,
        histogram_shift_prob=histogram_shift_prob,
        color_jitter_prob=color_jitter_prob,
        brightness=brightness,
        contrast=contrast,
        blur_probs=blur_probs,
        blur_sigma=blur_sigma,
        solarize_prob=solarize_prob,
        solarize_threshold=solarize_threshold,
        value_range=value_range,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
