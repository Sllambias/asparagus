import torch
import torch.nn.functional as F
from typing import Sequence


def antialiased_resize(image: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Bicubic resize of a (C, H, W) image or (C, D, H, W) volume, antialiased like PIL's, as torchvision's tensor resize.

    Args:
        image: Float image to resize.
        size: Output size per spatial axis.
    Returns:
        The resized image.
    """
    size = tuple(size)
    if image.ndim == 3:
        return F.interpolate(image.unsqueeze(0), size=size, mode="bicubic", antialias=True, align_corners=False).squeeze(0)
    # torch antialiases only in 2D, and the resize is separable: resize each slice (H, W), then along the depth
    channels, depth, height, width = image.shape
    slices = F.interpolate(
        image.reshape(channels * depth, 1, height, width), size=size[1:], mode="bicubic", antialias=True, align_corners=False
    )
    slices = slices.reshape(channels, 1, depth, size[1] * size[2])
    volume = F.interpolate(slices, size=(size[0], size[1] * size[2]), mode="bicubic", antialias=True, align_corners=False)
    return volume.reshape(channels, *size)
