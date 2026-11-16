"""Tests for RandomBlockMask, which follows the official DINOv2 masking policy."""

import pytest
import torch
from asparagus.modules.transforms.blockmask import RandomBlockMask


@pytest.mark.parametrize("grid_size", [(14, 14), (8, 8, 8)])
def test_half_of_the_views_are_masked_within_the_ratio_range(grid_size):
    batch_size = 64
    mask = RandomBlockMask(grid_size=grid_size)(batch_size=batch_size).flatten(1)
    ratios = mask.float().mean(1)

    assert mask.shape == (batch_size, torch.Size(grid_size).numel())
    assert int((ratios > 0).sum()) == batch_size // 2
    assert ratios.max() <= 0.5


def test_no_views_masked_with_zero_probability():
    mask = RandomBlockMask(grid_size=(14, 14), mask_probability=0.0)(batch_size=8)
    assert not mask.any()
