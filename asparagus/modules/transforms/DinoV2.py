import torch
import torch.nn as nn
from asparagus.functional.blur import gaussian_blur
from monai.transforms import (
    NormalizeIntensity,
    RandAffine,
    RandFlip,
    RandGaussianSmooth,
    RandHistogramShift,
    RandScaleCrop,
    Resize,
    ScaleIntensityRange,
    Transform,
)
from torchvision.transforms import Compose
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class RandomResizedCropND(Transform):
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
        mode: Optional[str] = None,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            prob (float): Probability of applying the random resized crop.
            size (Union[int, Tuple[int, int, int]]): Desired output size after resizing.
                If an int is provided, it will be used for all three dimensions.
            scale (Tuple[float, float]): Lower and upper bounds for the crop's share of the image area
             (volume in 3D) before resizing, as in torchvision's RandomResizedCrop.
            mode: Interpolation mode of the resize, bilinear in 2D and trilinear in 3D by default.
            clip_range: Intensity range to clip the resized crop to, as PIL does for uint8 images.
        """
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.size = size
        self.clip_range = clip_range
        if mode is not None:
            self.mode = mode
        elif len(size) == 3:
            self.mode = "trilinear"
        elif len(size) == 2:
            self.mode = "bilinear"
        else:
            print(f"Got size: {size} with len: {len(size)}. Expected length 2 or 3")

    def __call__(self, image):
        if torch.rand(1) < self.prob:
            # scale is a share of the area or volume, RandScaleCrop scales each axis
            random_scale = torch.empty(1).uniform_(*self.scale).item() ** (1 / len(self.size))
            image = RandScaleCrop(random_scale, random_size=False)(image)

            # Antialiased like the PIL resize in the official torchvision crop, with MONAI's and skimage's sigma.
            # MONAI's own anti-aliasing pads with zeros, which darkens the borders.
            sigma = [max(0.0, (in_size / out_size - 1) / 2) for in_size, out_size in zip(image.shape[1:], self.size)]
            if any(axis_sigma > 0 for axis_sigma in sigma):
                image = gaussian_blur(image, sigma, kernel_size=[2 * int(4 * axis_sigma + 0.5) + 1 for axis_sigma in sigma])
            image = Resize(self.size, mode=self.mode)(image)

            if self.clip_range is not None:
                image = image.clamp(*self.clip_range)

        return image


class RandomColorJitter(Transform):
    """
    Random brightness and contrast in random order, as torchvision's ColorJitter does on grayscale images,
    where saturation and hue have no effect.
    """

    def __init__(
        self,
        prob: float = 0.8,
        brightness: float = 0.4,
        contrast: float = 0.4,
        value_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            prob: Probability of applying the jitter.
            brightness: The brightness factor is drawn from [1 - brightness, 1 + brightness].
            contrast: The contrast factor is drawn from [1 - contrast, 1 + contrast].
            value_range: Intensity range of the image, which brightness scales from and both clip to.
                The image's own min and max are used when it is None.
        """
        super().__init__()
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast
        self.value_range = value_range

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) >= self.prob:
            return image
        if self.value_range is None:
            low, high = image.min(), image.max()
        else:
            low, high = self.value_range
        brightness = torch.empty(1).uniform_(max(0.0, 1 - self.brightness), 1 + self.brightness).item()
        contrast = torch.empty(1).uniform_(max(0.0, 1 - self.contrast), 1 + self.contrast).item()
        for adjustment in torch.randperm(2).tolist():
            if adjustment == 0:
                image = low + (image - low) * brightness
            else:
                mean = image.mean()
                image = mean + (image - mean) * contrast
            image = image.clamp(low, high)
        return image


class RandomGaussianBlur(Transform):
    """Isotropic Gaussian blur with a random sigma and a fixed kernel size, as torchvision's GaussianBlur, in 2D or 3D."""

    def __init__(self, prob: float = 0.5, sigma: Tuple[float, float] = (0.1, 2.0), kernel_size: int = 9):
        super().__init__()
        self.prob = prob
        self.sigma = sigma
        self.kernel_size = kernel_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) >= self.prob:
            return image
        sigma = torch.empty(1).uniform_(*self.sigma).item()
        spatial_dims = image.ndim - 1
        return gaussian_blur(image, [sigma] * spatial_dims, [self.kernel_size] * spatial_dims)


class RandomSolarize(Transform):
    """Inverts the intensities above a threshold, which is a share of the image's intensity range, as torchvision does."""

    def __init__(self, prob: float = 0.2, threshold: float = 0.5, value_range: Optional[Tuple[float, float]] = None):
        """
        Args:
            prob: Probability of solarizing.
            threshold: Share of the intensity range above which intensities are inverted.
            value_range: Intensity range of the image. The image's own min and max are used when it is None.
        """
        super().__init__()
        self.prob = prob
        self.threshold = threshold
        self.value_range = value_range

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.prob:
            if self.value_range is None:
                low, high = image.min(), image.max()
            else:
                low, high = self.value_range
            cutoff = low + self.threshold * (high - low)
            image = torch.where(image >= cutoff, high + low - image, image)
        return image


class DINOv2Augmentation(nn.Module):
    """
    2D/3D data augmentation for DINOv2: global and local views with crop, flip, affine, histogram shift,
    color jitter, blur and solarize. The defaults are the medical setup; the official setup turns affine and
    histogram shift off and color jitter, the official blur and normalization on.
    """

    def __init__(
        self,
        global_view_scale: Optional[List[float]] = None,
        global_view_size: Union[Tuple[int, int], Tuple[int, int, int]] = (48, 48, 48),
        local_view_scale: Optional[List[float]] = None,
        local_view_size: Union[Tuple[int, int], Tuple[int, int, int]] = (24, 24, 24),
        num_local_views: int = 2,
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
    ):
        """
        Initialize the DINOv2 augmentation pipeline.
        Args:
            global_view_scale: Scale range for global crops
            global_view_size: Output size for global crops
            local_view_scale: Scale range for local crops
            local_view_size: Output size for local crops
            num_local_views: Number of local views
            resize_mode: Interpolation of the crop resize, bilinear in 2D and trilinear in 3D by default
            flip_axis: Spatial axis of the random flip, left-right by default (the width in 2D, the first axis in 3D)
            affine_prob: Probability of a small random rotation and shear
            histogram_shift_prob: Probability of a random histogram shift
            color_jitter_prob: Probability of a random brightness and contrast jitter
            brightness: Brightness jitter strength
            contrast: Contrast jitter strength
            blur_probs: Blur probability of the first global view, the second global view and the local views
            blur_sigma: Sigma range of an isotropic kernel-9 blur, as official. MONAI's RandGaussianSmooth when None
            solarize_prob: Solarize probability of the second global view
            solarize_threshold: Share of the intensity range above which solarize inverts
            value_range: Input intensity range. When set, images are rescaled from it to [0, 1] before augmenting,
                as torchvision's ToTensor does, and the intensity transforms clip to [0, 1]
            normalize_mean: Mean subtracted from every view after augmenting
            normalize_std: Standard deviation every view is divided by after augmenting
        """
        super().__init__()
        if global_view_scale is None:
            global_view_scale = [0.32, 1.0]
        if local_view_scale is None:
            local_view_scale = [0.05, 0.32]
        self.global_view_scale = global_view_scale
        self.global_view_size = global_view_size
        self.local_view_scale = local_view_scale
        self.local_view_size = local_view_size
        self.num_local_views = num_local_views
        self.resize_mode = resize_mode
        self.flip_axis = flip_axis
        self.affine_prob = affine_prob
        self.histogram_shift_prob = histogram_shift_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.blur_sigma = blur_sigma
        self.solarize_threshold = solarize_threshold
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if value_range is None:
            self.rescale = None
            self.value_range = None
        else:
            low, high = value_range
            self.rescale = ScaleIntensityRange(a_min=low, a_max=high, b_min=0.0, b_max=1.0)
            self.value_range = (0.0, 1.0)

        if self.num_local_views == 0 and min(self.global_view_scale) > 0.4:
            self.global_view_scale[0] = sum(self.local_view_scale) / 2 if self.local_view_scale else 0.25

        global_blur_prob_1, global_blur_prob_2, local_blur_prob = blur_probs
        # The blur and solarize probabilities differ between the two global views and the local views, as official
        self.global_aug_1 = self.view_transforms(self.global_view_size, self.global_view_scale, blur_prob=global_blur_prob_1)
        self.global_aug_2 = self.view_transforms(
            self.global_view_size, self.global_view_scale, blur_prob=global_blur_prob_2, solarize_prob=solarize_prob
        )
        if self.num_local_views > 0:
            self.local_aug = self.view_transforms(self.local_view_size, self.local_view_scale, blur_prob=local_blur_prob)
        else:
            self.local_aug = None

    def view_transforms(
        self,
        size: Union[Tuple[int, int], Tuple[int, int, int]],
        scale: List[float],
        blur_prob: float,
        solarize_prob: float = 0.0,
    ) -> Compose:
        """Transforms for one view. Cropping first keeps the affine cheap."""
        # Official flips left-right: the width axis of a 2D image, the first axis of a RAS volume
        if self.flip_axis is not None:
            flip_axis = self.flip_axis
        elif len(size) == 2:
            flip_axis = 1
        else:
            flip_axis = 0

        if self.blur_sigma is None:
            blur = RandGaussianSmooth(prob=blur_prob)
        else:
            blur = RandomGaussianBlur(prob=blur_prob, sigma=self.blur_sigma)

        transforms = [
            RandomResizedCropND(prob=1, size=size, scale=scale, mode=self.resize_mode, clip_range=self.value_range),
            RandFlip(prob=0.5, spatial_axis=flip_axis),
            RandAffine(
                prob=self.affine_prob,
                rotate_range=(22 / 7) / 180 * 10,
                shear_range=0.1,
                padding_mode="zeros",
            ),
            RandHistogramShift(prob=self.histogram_shift_prob),
            RandomColorJitter(
                prob=self.color_jitter_prob,
                brightness=self.brightness,
                contrast=self.contrast,
                value_range=self.value_range,
            ),
            blur,
            RandomSolarize(prob=solarize_prob, threshold=self.solarize_threshold, value_range=self.value_range),
        ]
        if self.normalize_mean is not None or self.normalize_std is not None:
            assert self.normalize_mean is not None and self.normalize_std is not None, "Set both normalize_mean and _std"
            transforms.append(NormalizeIntensity(subtrahend=self.normalize_mean, divisor=self.normalize_std))
        return Compose(transforms)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply global and local augmentations to the image.
        Args:
            data: Sample dict with the image (C, H, W) or (C, D, H, W) under "image".
        Returns:
            The sample dict with the two global views under "image" and the local views under "local_views".
        """
        image = data["image"]
        if self.rescale is not None:
            image = self.rescale(image)
        if self.local_aug is not None:
            data["local_views"] = [self.local_aug(image.clone()) for _ in range(self.num_local_views)]
        data["image"] = [self.global_aug_1(image.clone()), self.global_aug_2(image.clone())]
        return data
