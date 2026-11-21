import torch
from monai.networks.layers import separable_filtering
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
    kernels = []
    for axis_sigma, size in zip(sigma, kernel_size):
        if size == 1:
            kernel = torch.ones(1)
        else:
            half_size = (size - 1) / 2
            kernel = torch.exp(-0.5 * (torch.linspace(-half_size, half_size, size) / axis_sigma) ** 2)
        kernels.append(kernel / kernel.sum())
    # MONAI's other Gaussian filters pad with zeros, which darkens the borders
    return separable_filtering(image.float().unsqueeze(0), kernels, mode="reflect").squeeze(0)
