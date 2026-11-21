"""DINOv2Augmentation runs the official DINOv2 augmentation on 8-bit 2D images and volumes; ultradino is UltraDino's."""

import itertools
import math
import multiprocessing
import numpy as np
import pytest
import torch
from asparagus.functional.resize import antialiased_resize
from asparagus.functional.seeding import seed_worker
from asparagus.modules.transforms import Torch_AsImage, Torch_GaussianBlur, Torch_RandomResizedCrop
from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from asparagus.modules.transforms.presets import dinov2, ultradino
from monai.transforms import RandAffine
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class RotatedConstantImage(Dataset):
    def __init__(self):
        self.rotate = RandAffine(prob=1.0, rotate_range=0.5)

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self.rotate(torch.rand(1, 32, 32, generator=torch.Generator().manual_seed(0)))


def test_forked_workers_draw_different_monai_parameters():
    loader = DataLoader(
        RotatedConstantImage(),
        batch_size=1,
        num_workers=2,
        multiprocessing_context=multiprocessing.get_context("fork"),
        worker_init_fn=seed_worker,
    )
    first_of_worker_0, first_of_worker_1 = list(loader)[:2]
    assert not torch.equal(first_of_worker_0, first_of_worker_1)


def random_8_bit_image(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))


SIZES = [((32, 48), (16, 16)), ((24, 32, 40), (8, 8, 8))]


@pytest.mark.parametrize("global_size, local_size", SIZES)
def test_views_have_the_requested_sizes(global_size, local_size):
    augmentation = DINOv2Augmentation(global_view_size=global_size, local_view_size=local_size, num_local_views=3)
    data = augmentation({"image": random_8_bit_image(1, *[2 * side for side in global_size])})
    assert [tuple(view.shape) for view in data["image"]] == [(1, *global_size)] * 2
    assert [tuple(view.shape) for view in data["local_views"]] == [(1, *local_size)] * 3


@pytest.mark.parametrize("global_size, local_size", SIZES)
def test_views_are_normalized_float_tensors(global_size, local_size):
    augmentation = DINOv2Augmentation(
        global_view_size=global_size, local_view_size=local_size, num_local_views=2, normalize_mean=[0.5], normalize_std=[0.25]
    )
    data = augmentation({"image": random_8_bit_image(1, *[2 * side for side in global_size])})
    for view in data["image"] + data["local_views"]:
        assert type(view) is torch.Tensor and view.dtype == torch.float32
        # The intensities are scaled to [0, 1], which the normalization maps to [-2, 2]
        assert -2.0 <= view.min() and view.max() <= 2.0


@pytest.mark.parametrize("global_size, local_size", SIZES)
def test_same_seed_gives_same_views(global_size, local_size):
    augmentation = DINOv2Augmentation(global_view_size=global_size, local_view_size=local_size, num_local_views=2)
    image = random_8_bit_image(1, *[2 * side for side in global_size])
    views = []
    for _ in range(2):
        torch.manual_seed(0)
        # Gardening Tools draws the crop corners with NumPy
        np.random.seed(0)
        data = augmentation({"image": image.clone()})
        views.append(data["image"] + data["local_views"])
    assert all(torch.equal(first, second) for first, second in zip(*views))


def test_no_local_views():
    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=0)
    assert "local_views" not in augmentation({"image": random_8_bit_image(1, 64, 64)})


@pytest.mark.parametrize(
    "global_size, image",
    [
        ((32, 32), torch.rand(1, 64, 64)),
        ((32, 32), torch.zeros(1, 16, 64, 64, dtype=torch.uint8)),
        ((16, 16, 16), torch.zeros(1, 64, 64, dtype=torch.uint8)),
    ],
)
def test_only_8_bit_images_of_the_view_dimensions_are_accepted(global_size, image):
    augmentation = DINOv2Augmentation(global_view_size=global_size, local_view_size=(8,) * len(global_size))
    with pytest.raises(ValueError, match="8-bit"):
        augmentation({"image": image})


def test_2d_crop_sizes_are_drawn_as_torchvision_draws_them():
    image = random_8_bit_image(1, 300, 400)
    for seed in range(50):
        torch.manual_seed(seed)
        sides = Torch_RandomResizedCrop.get_params(image.shape[1:], (0.05, 1.0), (math.log(3 / 4), math.log(4 / 3)))
        torch.manual_seed(seed)
        _, _, height, width = transforms.RandomResizedCrop.get_params(image, (0.05, 1.0), (3 / 4, 4 / 3))
        assert sides == [height, width]


@pytest.mark.parametrize("shape, largest_ratio", [((300, 400), 4 / 3), ((60, 80, 100), (4 / 3) ** 1.5)])
def test_crop_covers_the_requested_share_with_bounded_aspect_ratios(shape, largest_ratio):
    for scale in [(0.32, 1.0), (0.05, 0.32)]:
        log_ratio = (math.log(3 / 4), math.log(4 / 3))
        for _ in range(50):
            sides = Torch_RandomResizedCrop.get_params(shape, scale, log_ratio)
            # Rounding each side moves the share and the ratios a little
            assert 0.85 * scale[0] <= math.prod(sides) / math.prod(shape) <= 1.15 * scale[1]
            for first, second in itertools.combinations(sides, 2):
                assert max(first, second) / min(first, second) <= 1.1 * largest_ratio
            assert all(side <= size for side, size in zip(sides, shape))


@pytest.mark.parametrize("shape, size", [((1, 120, 160), (16, 16)), ((1, 30, 40, 50), (8, 8, 8))])
def test_crop_of_a_constant_image_stays_constant(shape, size):
    # The antialiasing and resize add no dark borders
    crop = Torch_RandomResizedCrop(size=size)
    for _ in range(10):
        cropped = crop({"image": torch.full(shape, 200, dtype=torch.uint8)})["image"]
        assert torch.equal(cropped, torch.full((1, *size), 200, dtype=torch.uint8))


def test_2d_resize_is_torchvisions_antialiased_bicubic():
    # torchvision's tensor resize is its match to PIL's
    image = random_8_bit_image(1, 300, 400)
    for size in [(224, 224), (96, 96), (400, 500)]:
        resized = antialiased_resize(image.float(), size).round().clamp(0, 255).to(torch.uint8)
        reference = TF.resize(image, list(size), transforms.InterpolationMode.BICUBIC, antialias=True)
        assert torch.equal(resized, reference)


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_flips_act_on_height_and_width(spatial_dims):
    image = random_8_bit_image(1, *(4, 6, 8)[-spatial_dims:])
    for vertical_flip, dim in [(False, -1), (True, -2)]:
        augmentation = DINOv2Augmentation(
            global_view_size=image.shape[1:], local_view_size=(4,) * spatial_dims, vertical_flip=vertical_flip
        )
        flip = augmentation.geometric_augmentation_global.transforms[1]
        outputs = {tuple(flip({"image": image.clone()})["image"].flatten().tolist()) for _ in range(20)}
        assert outputs == {tuple(image.flatten().tolist()), tuple(image.flip(dim).flatten().tolist())}


def test_2d_blur_is_torchvisions():
    image = random_8_bit_image(1, 40, 56)
    for seed in range(5):
        torch.manual_seed(seed)
        blurred = Torch_GaussianBlur(p_per_sample=1.0)({"image": image.clone()})["image"]
        torch.manual_seed(seed)
        reference = transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=1.0)(image)
        # Rounding to 8 bits can differ by one where the two blurs differ in float rounding
        assert (blurred.int() - reference.int()).abs().max() <= 1


def depth_constant_volume():
    image = random_8_bit_image(1, 40, 56)
    return image, image.unsqueeze(1).repeat(1, 12, 1, 1)


def test_a_depth_constant_volume_is_blurred_like_its_slices():
    image, volume = depth_constant_volume()
    torch.manual_seed(0)
    blurred_volume = Torch_GaussianBlur(p_per_sample=1.0)({"image": volume})["image"]
    torch.manual_seed(0)
    blurred_image = Torch_GaussianBlur(p_per_sample=1.0)({"image": image})["image"]
    assert (blurred_volume.int() - blurred_image.int().unsqueeze(1)).abs().max() <= 1


def test_a_depth_constant_volume_is_jittered_like_its_slices():
    image, volume = depth_constant_volume()
    jitter = Torch_AsImage(transforms.ColorJitter(0.4, 0.4, 0.2, 0.1))
    for seed in range(5):
        torch.manual_seed(seed)
        jittered_volume = jitter({"image": volume})["image"]
        torch.manual_seed(seed)
        jittered_image = jitter({"image": image})["image"]
        assert torch.equal(jittered_volume, jittered_image.unsqueeze(1).expand_as(volume))


def test_a_depth_constant_volume_is_resized_like_its_slices():
    image, volume = depth_constant_volume()
    resized_volume = antialiased_resize(volume.float(), (6, 20, 24))
    resized_image = antialiased_resize(image.float(), (20, 24))
    assert torch.allclose(resized_volume, resized_image.unsqueeze(1).expand_as(resized_volume), atol=1e-3)


def describe(compose):
    """The transforms of a pipeline, with the ones Torch_AsImage and RandomApply wrap."""
    names = []
    for transform in compose.transforms:
        if isinstance(transform, Torch_AsImage):
            transform = transform.transform
        if isinstance(transform, transforms.RandomApply):
            transform = transform.transforms[0]
        names.append(type(transform).__name__)
    return names


@pytest.mark.parametrize("global_size, local_size", SIZES)
def test_ultradino_changes_the_flip_jitter_grayscale_and_normalization(global_size, local_size):
    official = dinov2(global_view_size=global_size, local_view_size=local_size)
    ultra = ultradino(global_view_size=global_size, local_view_size=local_size)
    width_axis, height_axis = len(global_size) - 1, len(global_size) - 2
    assert official.geometric_augmentation_global.transforms[1].axes == (width_axis,)
    assert ultra.geometric_augmentation_global.transforms[1].axes == (height_axis,)
    assert describe(official.global_transfo2) == [
        "ColorJitter",
        "RandomGrayscale",
        "Torch_GaussianBlur",
        "RandomSolarize",
        "ConvertImageDtype",
        "Normalize",
    ]
    assert describe(ultra.global_transfo2) == [
        "ColorJitter",
        "Torch_GaussianBlur",
        "RandomSolarize",
        "ConvertImageDtype",
        "Normalize",
    ]
    assert official.global_transfo2.transforms[0].transform.transforms[0].hue == (-0.1, 0.1)
    assert ultra.global_transfo2.transforms[0].transform.transforms[0].hue is None
    assert ultra.global_transfo2.transforms[-1].transform.mean == [0.1842924807]
