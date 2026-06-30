import torch
import torch.nn as nn
from monai.transforms import (
    RandAffine,
    RandGaussianSmooth,
    RandHistogramShift,
    RandScaleCrop,
    Resize,
    SpatialPad,
    Transform,
)
from torchvision.transforms import Compose
from typing import List, Optional, Tuple, Union


class RandomResizedCrop3D(Transform):
    """
    Combines monai's random spatial crop followed by resize to the desired size.

    Modification:
    1. The spatial crop is done with same dimensions for all the axes
    2. Handles cases where the image_size is less than the crop_size by choosing
        the smallest dimension as the random scale.

    """

    def __init__(
        self,
        prob: float = 1,
        size: Union[Tuple[int, int], Tuple[int, int, int]] = (50, 50, 50),
        scale: Tuple[float, float] = (0.5, 1.0),
    ):
        """
        Args:
            prob (float): Probability of applying the random resized crop.
            size (Union[int, Tuple[int, int, int]]): Desired output size after resizing.
                If an int is provided, it will be used for all three dimensions.
            scale (List[int]): Specifies the lower and upper bounds for the random area of the crop,
             before resizing. The scale is defined with respect to the area of the original image.
        """
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.size = size
        if len(size) == 3:
            self.mode = "trilinear"
        elif len(size) == 2:
            self.mode = "bilinear"
        else:
            print(f"Got size: {size} with len: {len(size)}. Expected length 2 or 3")

    def __call__(self, image):
        if torch.rand(1) < self.prob:
            random_scale = torch.empty(1).uniform_(*self.scale).item()
            rand_cropper = RandScaleCrop(random_scale, random_size=False)
            resizer = Resize(self.size, mode=self.mode)

            for transform in [rand_cropper, resizer]:
                image = transform(image)

        return image


class DINOv2Augmentation(nn.Module):
    """
    3D data augmentation for DINOv2: global and local views, affine, histogram, smoothing, cropping.
    """

    def __init__(
        self,
        global_view_scale: Optional[List[float]] = None,
        global_view_size: Union[int, Tuple[int, int, int]] = 48,
        local_view_scale: Optional[List[float]] = None,
        local_view_size: Union[int, Tuple[int, int, int]] = 24,
        num_local_views: int = 2,
    ):
        """
        Initialize 3D DINOv2 augmentation pipeline.
        Args:
            global_view_scale: Scale range for global crops
            global_view_size: Output size for global crops
            local_view_scale: Scale range for local crops
            local_view_size: Output size for local crops
            num_local_views: Number of local views
        """
        super().__init__()
        if global_view_scale is None:
            global_view_scale = [0.3, 1.0]
        if local_view_scale is None:
            local_view_scale = [0.1, 0.3]
        self.global_view_scale = global_view_scale
        self.global_view_size = global_view_size
        self.local_view_scale = local_view_scale
        self.local_view_size = local_view_size
        self.num_local_views = num_local_views

        if self.num_local_views == 0 and min(self.global_view_scale) > 0.4:
            self.global_view_scale[0] = sum(self.local_view_scale) / 2 if self.local_view_scale else 0.25

        self.global_aug = Compose(
            [
                RandAffine(
                    prob=0.5,
                    rotate_range=(22 / 7) / 180 * 10,
                    shear_range=0.1,
                    padding_mode="zeros",
                ),
                RandomResizedCrop3D(prob=1, size=self.global_view_size, scale=self.global_view_scale),
                RandHistogramShift(prob=0.5),
                RandGaussianSmooth(prob=0.5),
                SpatialPad(spatial_size=self.global_view_size),
            ]
        )
        if self.num_local_views > 0:
            self.local_aug = Compose(
                [
                    RandAffine(
                        prob=0.5,
                        rotate_range=(22 / 7) / 180 * 10,
                        shear_range=0.1,
                        padding_mode="zeros",
                    ),
                    RandomResizedCrop3D(prob=1, size=self.local_view_size, scale=self.local_view_scale),
                    RandHistogramShift(prob=0.5),
                    RandGaussianSmooth(prob=0.5),
                    SpatialPad(spatial_size=self.local_view_size),
                ]
            )
        else:
            self.local_aug = None

    def forward(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply global and local augmentations to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W).
        Returns:
            List[torch.Tensor]: List containing N augmented tensors for global (2) and local (N-2) views.
        """
        if self.local_aug is not None:
            data["local_views"] = [self.local_aug(data["image"].clone()) for _ in range(self.num_local_views)]
        data["image"] = [self.global_aug(data["image"].clone()) for _ in range(2)]
        return data


# %%

# %%
