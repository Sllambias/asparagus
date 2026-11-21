import torch
from asparagus.functional.blur import gaussian_blur
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from typing import Any, Dict, Tuple


class Torch_GaussianBlur(BaseTransform):
    """torchvision's GaussianBlur for 2D images (C, H, W) and volumes (C, D, H, W): a kernel of fixed size, one random
    sigma for all axes and reflect padding. Integer images are rounded back, as torchvision does."""

    def __init__(
        self,
        p_per_sample: float = 1.0,
        kernel_size: int = 9,
        sigma: Tuple[float, float] = (0.1, 2.0),
        data_key: str = "image",
    ) -> None:
        self.p_per_sample = p_per_sample
        self.kernel_size = kernel_size
        self.sigma = tuple(sigma)
        self.data_key = data_key

    @staticmethod
    def get_params(sigma: Tuple[float, float]) -> float:
        return torch.empty(1).uniform_(sigma[0], sigma[1]).item()

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if torch.rand(1) < self.p_per_sample:
            image = data_dict[self.data_key]
            spatial_dims = image.ndim - 1
            sigma = self.get_params(self.sigma)
            blurred = gaussian_blur(image, [sigma] * spatial_dims, [self.kernel_size] * spatial_dims)
            if not image.is_floating_point():
                blurred = blurred.round().to(image.dtype)
            data_dict[self.data_key] = blurred
        return data_dict
