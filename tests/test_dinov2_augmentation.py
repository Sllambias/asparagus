"""The DINOv2 crop scale is a share of the image area (volume in 3D), as in torchvision's RandomResizedCrop."""

import math
import multiprocessing
import pytest
import torch
import torchvision.transforms.functional as TF
from asparagus.functional.seeding import seed_worker
from asparagus.modules.transforms import DinoV2
from asparagus.modules.transforms.DinoV2 import (
    DINOv2Augmentation,
    RandomColorJitter,
    RandomGaussianBlur,
    RandomResizedCropND,
)
from monai.transforms import RandAffine
from torch.utils.data import DataLoader, Dataset


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


@pytest.mark.parametrize("global_size, local_size", [((32, 48), (16, 16)), ((24, 32, 40), (8, 8, 8))])
def test_views_have_the_requested_sizes(global_size, local_size):
    augmentation = DINOv2Augmentation(global_view_size=global_size, local_view_size=local_size, num_local_views=3)
    data = augmentation({"image": torch.rand(1, *[2 * side for side in global_size])})
    assert [tuple(view.shape[1:]) for view in data["image"]] == [global_size] * 2
    assert [tuple(view.shape[1:]) for view in data["local_views"]] == [local_size] * 3


def test_views_are_plain_tensors():
    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=2)
    data = augmentation({"image": torch.rand(1, 64, 64)})
    assert all(type(view) is torch.Tensor for view in data["image"] + data["local_views"])


def test_blur_matches_torchvision():
    image = torch.rand(1, 40, 56)
    torch.manual_seed(1)
    blurred = RandomGaussianBlur(prob=1.0, sigma=(0.1, 2.0))(image)
    # Replay the random draws: the probability, then sigma
    torch.manual_seed(1)
    torch.rand(1)
    sigma = torch.empty(1).uniform_(0.1, 2.0).item()
    assert torch.allclose(blurred, TF.gaussian_blur(image, kernel_size=9, sigma=sigma), atol=1e-6)


@pytest.mark.parametrize("shape", [(1, 20, 24), (1, 12, 14, 16)])
def test_blur_keeps_shape_and_constant_images(shape):
    image = torch.full(shape, 0.3)
    blurred = RandomGaussianBlur(prob=1.0)(image)
    assert blurred.shape == image.shape
    assert torch.allclose(blurred, image)


def test_color_jitter_matches_torchvision():
    image = torch.rand(1, 40, 56)
    orders = set()
    for seed in range(10):
        torch.manual_seed(seed)
        jittered = RandomColorJitter(prob=1.0, value_range=(0.0, 1.0))(image)
        # Replay the random draws: the probability, the two factors, then the order
        torch.manual_seed(seed)
        torch.rand(1)
        brightness = torch.empty(1).uniform_(0.6, 1.4).item()
        contrast = torch.empty(1).uniform_(0.6, 1.4).item()
        order = torch.randperm(2).tolist()
        expected = image
        for adjustment in order:
            if adjustment == 0:
                expected = TF.adjust_brightness(expected, brightness)
            else:
                expected = TF.adjust_contrast(expected, contrast)
        assert torch.allclose(jittered, expected, atol=1e-6)
        orders.add(tuple(order))
    assert len(orders) == 2


@pytest.mark.parametrize("global_size, local_size", [((32, 48), (16, 16)), ((24, 32, 40), (8, 8, 8))])
def test_official_views_stay_in_the_unit_range(global_size, local_size):
    augmentation = DINOv2Augmentation(
        global_view_size=global_size,
        local_view_size=local_size,
        num_local_views=2,
        resize_mode="bicubic" if len(global_size) == 2 else None,
        affine_prob=0.0,
        histogram_shift_prob=0.0,
        color_jitter_prob=1.0,
        blur_probs=(1.0, 1.0, 1.0),
        blur_sigma=(0.1, 2.0),
        solarize_prob=1.0,
        value_range=(0, 255),
    )
    image = torch.randint(0, 256, (1, *[2 * side for side in global_size]), dtype=torch.uint8)
    data = augmentation({"image": image})
    for view in data["image"] + data["local_views"]:
        assert 0.0 <= view.min() and view.max() <= 1.0


def test_views_are_normalized():
    augmentation = DINOv2Augmentation(
        global_view_size=(32, 32),
        local_view_size=(16, 16),
        num_local_views=1,
        affine_prob=0.0,
        histogram_shift_prob=0.0,
        blur_probs=(0.0, 0.0, 0.0),
        solarize_prob=0.0,
        value_range=(0, 255),
        normalize_mean=0.5,
        normalize_std=0.25,
    )
    data = augmentation({"image": torch.full((1, 64, 64), 255, dtype=torch.uint8)})
    # Constant views show that the antialiasing does not darken the borders
    for view in data["image"] + data["local_views"]:
        assert torch.allclose(view, torch.full_like(view, 2.0))
