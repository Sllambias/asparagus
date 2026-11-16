"""The DINOv2 crop scale is a share of the image area (volume in 3D), as in torchvision's RandomResizedCrop."""

import math
import pytest
import torch
from asparagus.modules.transforms import DinoV2
from asparagus.modules.transforms.DinoV2 import RandomResizedCropND


@pytest.mark.parametrize("shape", [(1, 100, 100), (1, 40, 40, 40)])
@pytest.mark.parametrize("scale", [(0.32, 1.0), (0.05, 0.32)])
def test_crop_covers_the_requested_share_of_the_image(monkeypatch, shape, scale):
    # Skip the resize so the output shape is the crop shape
    monkeypatch.setattr(DinoV2, "Resize", lambda size, mode: lambda image: image)
    crop = RandomResizedCropND(size=shape[1:], scale=scale)
    image = torch.zeros(shape)
    # MONAI rounds each side of the crop up
    largest_share = math.prod(math.ceil(scale[1] ** (1 / len(shape[1:])) * side) for side in shape[1:]) / image.numel()
    for _ in range(50):
        share = crop(image).numel() / image.numel()
        assert scale[0] <= share <= largest_share
