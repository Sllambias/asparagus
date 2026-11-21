import math
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from torch import nn
from typing import Any, Dict


class Torch_AsImage(BaseTransform):
    """Applies a torchvision image transform to the image of a sample dict. A volume (C, D, H, W) is viewed as the image
    (C, D * H, W), which suits transforms that change each voxel alike from whole-image statistics, such as ColorJitter."""

    def __init__(self, transform: nn.Module, data_key: str = "image") -> None:
        self.transform = transform
        self.data_key = data_key

    @staticmethod
    def get_params() -> None:
        pass

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        image = data_dict[self.data_key]
        channels, *spatial = image.shape
        transformed = self.transform(image.reshape(channels, math.prod(spatial[:-1]), spatial[-1]))
        data_dict[self.data_key] = transformed.reshape(transformed.shape[0], *spatial)
        return data_dict
