import torch
import torch.nn.functional as F
from typing import Sequence


def gaussian_blur(image: torch.Tensor, sigma: Sequence[float], kernel_size: Sequence[int]) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) or (C, D, H, W) image with reflect padding, as torchvision's gaussian_blur.

    Args:
        image: Image to blur.
        sigma: Standard deviation per spatial axis.
        kernel_size: Odd kernel size per spatial axis. Axes with size 1 are not blurred.
    Returns:
        The blurred image as float.
    """
    channels = image.shape[0]
    spatial_dims = image.ndim - 1
    conv = F.conv2d if spatial_dims == 2 else F.conv3d
    # F.pad takes the padding of the last axis first
    padding = []
    for size in reversed(kernel_size):
        padding += [size // 2, size // 2]
    blurred = F.pad(image.float().unsqueeze(0), padding, mode="reflect")

    for axis, (axis_sigma, size) in enumerate(zip(sigma, kernel_size)):
        if size == 1:
            continue
        half_size = (size - 1) / 2
        kernel = torch.exp(-0.5 * (torch.linspace(-half_size, half_size, size) / axis_sigma) ** 2)
        kernel = kernel / kernel.sum()
        shape = [1] * spatial_dims
        shape[axis] = size
        weight = kernel.reshape(1, 1, *shape).repeat(channels, 1, *([1] * spatial_dims))
        blurred = conv(blurred, weight.to(blurred.device), groups=channels)
    return blurred.squeeze(0)
