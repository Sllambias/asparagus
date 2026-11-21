import torch
import torch.nn as nn
from asparagus.modules.transforms.as_image import Torch_AsImage
from asparagus.modules.transforms.gaussian_blur import Torch_GaussianBlur
from asparagus.modules.transforms.random_resized_crop import Torch_RandomResizedCrop
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from torchvision import transforms
from typing import Any, Dict, Sequence, Tuple, Union


class DINOv2Augmentation(nn.Module):
    """
    The official DINOv2 multi-crop augmentation (DataAugmentationDINO in dinov2/data/augmentations.py) for 8-bit 2D images
    (C, H, W) and volumes (C, D, H, W). Both go through the same transforms: the official ones, from Gardening Tools,
    torchvision or asparagus where torchvision's work only in 2D. The flips act on H and W.
    """

    def __init__(
        self,
        global_view_size: Union[Tuple[int, int], Tuple[int, int, int]] = (224, 224),
        local_view_size: Union[Tuple[int, int], Tuple[int, int, int]] = (96, 96),
        global_view_scale: Tuple[float, float] = (0.32, 1.0),
        local_view_scale: Tuple[float, float] = (0.05, 0.32),
        num_local_views: int = 8,
        vertical_flip: bool = False,
        hue: float = 0.1,
        grayscale_prob: float = 0.2,
        normalize_mean: Sequence[float] = (0.449,),
        normalize_std: Sequence[float] = (0.226,),
    ) -> None:
        """
        Args:
            global_view_size: Size of the global crops, (H, W) or (D, H, W).
            local_view_size: Size of the local crops, likewise.
            global_view_scale: Range of the global crops' share of the image area or volume.
            local_view_scale: Range of the local crops' share of the image area or volume.
            num_local_views: Number of local crops.
            vertical_flip: Flip up-down instead of left-right.
            hue: Hue jitter strength. Hue does nothing to grayscale images, but the jitter draws it.
            grayscale_prob: Probability of RandomGrayscale, left out at 0.
            normalize_mean: Mean per channel. The default is ImageNet's, averaged over RGB for grayscale images.
            normalize_std: Standard deviation per channel, likewise.
        """
        super().__init__()
        self.spatial_dims = len(global_view_size)
        self.num_local_views = num_local_views
        flip_axis = self.spatial_dims - 2 if vertical_flip else self.spatial_dims - 1
        flip = Torch_Mirror(p_per_sample=1.0, axes=(flip_axis,), p_mirror_per_axis=0.5)

        self.geometric_augmentation_global = transforms.Compose(
            [Torch_RandomResizedCrop(global_view_size, scale=global_view_scale), flip]
        )
        self.geometric_augmentation_local = transforms.Compose(
            [Torch_RandomResizedCrop(local_view_size, scale=local_view_scale), flip]
        )

        color_jittering = [
            Torch_AsImage(
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=hue)],
                    p=0.8,
                )
            )
        ]
        if grayscale_prob > 0:
            color_jittering.append(Torch_AsImage(transforms.RandomGrayscale(p=grayscale_prob)))
        normalize = [
            Torch_AsImage(transforms.ConvertImageDtype(torch.float32)),
            Torch_AsImage(transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std))),
        ]

        # Official GaussianBlur(p) blurs with probability 1 - p, so its p = 1.0, 0.1 and 0.5 blur 0%, 90% and 50% of the views
        self.global_transfo1 = transforms.Compose(color_jittering + [Torch_GaussianBlur(p_per_sample=0.0)] + normalize)
        self.global_transfo2 = transforms.Compose(
            color_jittering
            + [Torch_GaussianBlur(p_per_sample=0.9), Torch_AsImage(transforms.RandomSolarize(threshold=128, p=0.2))]
            + normalize
        )
        self.local_transfo = transforms.Compose(color_jittering + [Torch_GaussianBlur(p_per_sample=0.5)] + normalize)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data: Sample dict with a uint8 image (C, H, W) or volume (C, D, H, W) under "image".
        Returns:
            The sample dict with the two global views under "image" and the local views under "local_views".
        """
        image = data["image"]
        if image.dtype != torch.uint8 or image.ndim != self.spatial_dims + 1:
            raise ValueError(
                f"DINOv2Augmentation takes 8-bit {self.spatial_dims}D images with a channel axis, as official, "
                f"got {image.dtype} {tuple(image.shape)}"
            )
        data["image"] = [
            self.global_transfo1(self.geometric_augmentation_global({"image": image}))["image"],
            self.global_transfo2(self.geometric_augmentation_global({"image": image}))["image"],
        ]
        if self.num_local_views > 0:
            data["local_views"] = [
                self.local_transfo(self.geometric_augmentation_local({"image": image}))["image"]
                for _ in range(self.num_local_views)
            ]
        return data
