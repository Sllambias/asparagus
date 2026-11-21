import math
import torch
from gardening_tools.functional.transforms.cropping_and_padding import torch_croppad
from gardening_tools.functional.transforms.sampling import torch_resize
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from typing import Any, Dict, List, Sequence, Tuple


class Torch_RandomResizedCrop(BaseTransform):
    """torchvision's RandomResizedCrop for 2D images (C, H, W) and volumes (C, D, H, W) of 8-bit intensities: a crop with
    a random share of the area or volume and log-uniform aspect ratios between neighbouring sides, resized to size.
    Gardening Tools cuts the crop and resizes it, bicubic and antialiased like PIL's."""

    def __init__(
        self,
        size: Sequence[int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        data_key: str = "image",
    ) -> None:
        self.size = tuple(size)
        self.scale = tuple(scale)
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        self.data_key = data_key

    @staticmethod
    def get_params(shape: Sequence[int], scale: Tuple[float, float], log_ratio: Tuple[float, float]) -> List[int]:
        """Sides of the crop, drawn as torchvision draws them in 2D: 10 tries, then the whole image."""
        area = math.prod(shape)
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            # In 2D this is torchvision's width / height ratio; in 3D the neighbouring sides get one ratio each
            log_offsets = [0.0] * len(shape)
            for axis in range(len(shape) - 1):
                log_aspect = torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).item()
                log_offsets[axis] -= log_aspect / 2
                log_offsets[axis + 1] += log_aspect / 2
            sides = [int(round(target_area ** (1 / len(shape)) * math.exp(offset))) for offset in log_offsets]
            if all(0 < side <= size for side, size in zip(sides, shape)):
                return sides
        return list(shape)

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        image = data_dict[self.data_key]
        sides = self.get_params(image.shape[1:], self.scale, self.log_ratio)
        # The crop always fits, so Gardening Tools only picks a random corner and cuts
        crop, _ = torch_croppad(
            image,
            patch_size=sides,
            input_dims=len(sides),
            target_image_shape=(image.shape[0], *sides),
            target_label_shape=(1, *sides),
            mode="constant",
            value=0,
        )
        resized = torch_resize(crop, target_size=self.size, clip_to_input_range=False, antialias=True, mode="bicubic")
        # Saturated at 8 bits, as PIL does with the bicubic overshoot
        data_dict[self.data_key] = resized.round().clamp(0, 255).to(image.dtype)
        return data_dict
