"""
RandomBlockMask3D: 3D block masking for self-supervised learning (DINOv2-style).
Supports simple and advanced block masking for volumetric data.
"""

import numpy as np
import torch
from torch import Tensor, nn
from typing import Optional, Tuple, Union


def random_block_mask(grid_size: Union[Tuple[int, int], Tuple[int, int, int]], sequence_length: int):
    num_patches = sequence_length - 1
    assert np.prod(grid_size).item() == num_patches, (
        f"Grid size {np.prod(grid_size).item()} ({grid_size}) does not match patch count {num_patches}"
    )

    if len(grid_size) == 2:
        return RandomBlockMask2D(grid_size=grid_size, max_block_size=3)
    elif len(grid_size) == 3:
        return RandomBlockMask3D(grid_size=grid_size, max_block_size=3)

    # block_mask = block_masker(size=(B, D, H, W), device=device)
    # patch_mask = block_mask.flatten(start_dim=1)
    # mask = torch.zeros((B, sequence_length), device=device, dtype=torch.bool)
    # mask[:, 1:] = patch_mask


class RandomBlockMask3D(nn.Module):
    """
    A class for generating 3D random block masks for self-supervised learning.
    Supports both simple and advanced (multi-block, target ratio) masking.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int, int],
        ratio_min: float = 0.1,
        ratio_max: float = 0.5,
        min_block_size: int = 1,
        max_block_size: Optional[int] = None,
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.25),
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
    ):
        """
        Initialize the 3D random block masking.
        Args: see class docstring for details.
        """
        super().__init__()
        self.grid_size = grid_size
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.aspect_ratio_range = aspect_ratio_range
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.mode = "advanced"

    def forward(
        self,
        batch_size,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """
        Generate 3D random block masks for a batch.
        Args:
            size: Tensor size as (batch_size, depth, height, width)
            device: Device to create tensors on
        Returns:
            Boolean tensor of shape (batch_size, depth, height, width) where True indicates masked patches
        """
        if self.mode == "advanced":
            # Use advanced block masking strategy
            return self.advanced_block_mask(batch_size, mask_ratio=0.75, device=device)
        else:
            # Default to simple block masking
            return self.simple_block_mask(batch_size, device)

    def _calculate_block_sizes(self, spatial_dims):
        """Calculate block sizes for all dimensions efficiently."""
        ratio = torch.empty(1).uniform_(self.ratio_min, self.ratio_max).item()
        sizes = [max(self.min_block_size, int(dim * ratio)) for dim in spatial_dims]

        # Apply constraints
        if self.max_block_size is not None:
            sizes = [min(size, self.max_block_size) for size in sizes]

        # Ensure sizes don't exceed grid dimensions
        return [min(size, dim) for size, dim in zip(sizes, spatial_dims)]

    def simple_block_mask(
        self,
        batch_size: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """Generate simple, single, 3D block masks using ratio-based sizing."""
        B = batch_size
        size = (B, *self.grid_size)
        D, H, W = self.grid_size

        # Calculate block sizes efficiently
        block_sizes = self._calculate_block_sizes(self.grid_size)

        # Calculate valid placement ranges and generate random starts
        valid_ranges = [max(1, dim - block_size + 1) for dim, block_size in zip(self.grid_size, block_sizes)]
        starts = [torch.randint(0, valid_range, (B,), device=device) for valid_range in valid_ranges]

        # Create masks
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        for i in range(B):
            slices = [slice(start[i], start[i] + block_size) for start, block_size in zip(starts, block_sizes)]
            if len(self.grid_size) == 2:
                mask[i, slices[0], slices[1]] = True
            elif len(self.grid_size) == 3:
                mask[i, slices[0], slices[1], slices[2]] = True
        return mask

    def advanced_block_mask(
        self,
        batch_size,
        mask_ratio: float = 0.75,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """
        Generate advanced 3D block masks with target masking ratio.

        This method creates multiple blocks per image (volume patches positions) to reach a target
        masking ratio, similar to DINOv2's strategy but adapted for 3D.

        Args:
            size: Tensor size as (batch_size, depth, height, width)
            mask_ratio: Target ratio of patches to mask
            device: Device to create tensors on

        Returns:
            Boolean tensor where True indicates masked patches
        """
        B = batch_size
        D, H, W = self.grid_size
        total_patches = D * H * W
        target_masked = int(total_patches * mask_ratio)

        masks = []
        for batch_idx in range(B):
            mask = torch.zeros((D, H, W), dtype=torch.bool, device=device)
            current_masked = 0

            for _ in range(50):  # Max attempts to prevent infinite loops
                if current_masked >= target_masked:
                    break

                remaining = target_masked - current_masked
                target_patches = self._get_target_patches(remaining)

                # Calculate 3D block dimensions from target patches
                block_dims = self._calculate_3d_block_dims(target_patches)
                block_dims = self._apply_size_constraints(block_dims)

                # Try to place block
                if self._try_place_block(mask, block_dims, D, H, W):
                    new_masked = block_dims[0] * block_dims[1] * block_dims[2]
                    current_masked += new_masked

            masks.append(mask)

        return torch.stack(masks)

    def _get_target_patches(self, remaining):
        """Get target number of patches for next block."""
        if self.num_masking_patches is not None:
            return min(remaining, self.num_masking_patches)
        return max(self.min_num_patches, min(remaining, self.max_num_patches or remaining))

    def _calculate_3d_block_dims(self, target_patches):
        """Calculate 3D block dimensions from target patch count."""
        import random

        base_size = max(1, int(target_patches ** (1 / 3)))
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        return [
            max(1, target_patches // (base_size * base_size)),  # depth
            max(1, int(base_size * aspect_ratio)),  # height
            max(1, int(base_size / aspect_ratio)),  # width
        ]

    def _apply_size_constraints(self, block_dims):
        """Apply size constraints to block dimensions."""

        # Apply minimum size constraint
        block_dims = [max(self.min_block_size, dim) for dim in block_dims]

        # Apply maximum size constraint
        if self.max_block_size is not None:
            block_dims = [min(dim, self.max_block_size) for dim in block_dims]

        # Ensure blocks fit within volume
        return [min(block_dim, vol_dim) for block_dim, vol_dim in zip(block_dims, self.grid_size)]

    def _try_place_block(self, mask, block_dims, D, H, W):
        """Try to place a block in the mask. Returns True if successful."""
        import random

        # Calculate valid placement ranges
        valid_ranges = [max(1, vol_dim - block_dim + 1) for vol_dim, block_dim in zip([D, H, W], block_dims)]

        if any(valid_range <= 0 for valid_range in valid_ranges):
            return False

        # Random placement
        starts = [random.randint(0, valid_range - 1) for valid_range in valid_ranges]
        ends = [start + block_dim for start, block_dim in zip(starts, block_dims)]

        # Check if placement adds new masked patches
        slices = [slice(start, end) for start, end in zip(starts, ends)]
        block_region = mask[slices[0], slices[1], slices[2]]

        if (~block_region).sum().item() > 0:
            mask[slices[0], slices[1], slices[2]] = True
            return True

        return False


class RandomBlockMask2D(RandomBlockMask3D):
    """
    A class for generating 3D random block masks for self-supervised learning.
    Supports both simple and advanced (multi-block, target ratio) masking.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        ratio_min: float = 0.1,
        ratio_max: float = 0.5,
        min_block_size: int = 1,
        max_block_size: Optional[int] = None,
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.25),
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
    ):
        """
        Initialize the 3D random block masking.
        Args: see class docstring for details.
        """
        super().__init__(
            grid_size=grid_size,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            aspect_ratio_range=aspect_ratio_range,
            num_masking_patches=num_masking_patches,
            min_num_patches=min_num_patches,
            max_num_patches=max_num_patches,
        )

    def advanced_block_mask(
        self,
        batch_size,
        mask_ratio: float = 0.75,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """
        Generate advanced 3D block masks with target masking ratio.

        This method creates multiple blocks per image (volume patches positions) to reach a target
        masking ratio, similar to DINOv2's strategy but adapted for 3D.

        Args:
            size: Tensor size as (batch_size, depth, height, width)
            mask_ratio: Target ratio of patches to mask
            device: Device to create tensors on

        Returns:
            Boolean tensor where True indicates masked patches
        """
        B = batch_size
        H, W = self.grid_size
        total_patches = H * W
        target_masked = int(total_patches * mask_ratio)

        masks = []
        for batch_idx in range(B):
            mask = torch.zeros((H, W), dtype=torch.bool, device=device)
            current_masked = 0

            for _ in range(50):  # Max attempts to prevent infinite loops
                if current_masked >= target_masked:
                    break

                remaining = target_masked - current_masked
                target_patches = self._get_target_patches(remaining)

                # Calculate 2D block dimensions from target patches
                block_dims = self._calculate_2d_block_dims(target_patches)
                block_dims = self._apply_size_constraints(block_dims)

                # Try to place block
                if self._try_place_block(mask, block_dims, H, W):
                    new_masked = block_dims[0] * block_dims[1]
                    current_masked += new_masked

            masks.append(mask)

        return torch.stack(masks)

    def _calculate_2d_block_dims(self, target_patches):
        """Calculate 2D block dimensions from target patch count."""
        import random

        base_size = max(1, int(target_patches ** (1 / 2)))
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        return [
            max(1, int(base_size * aspect_ratio)),  # height
            max(1, int(base_size / aspect_ratio)),  # width
        ]

    def _try_place_block(self, mask, block_dims, H, W):
        """Try to place a block in the mask. Returns True if successful."""
        import random

        # Calculate valid placement ranges
        valid_ranges = [max(1, vol_dim - block_dim + 1) for vol_dim, block_dim in zip([H, W], block_dims)]

        if any(valid_range <= 0 for valid_range in valid_ranges):
            return False

        # Random placement
        starts = [random.randint(0, valid_range - 1) for valid_range in valid_ranges]
        ends = [start + block_dim for start, block_dim in zip(starts, block_dims)]

        # Check if placement adds new masked patches
        slices = [slice(start, end) for start, end in zip(starts, ends)]
        block_region = mask[slices[0], slices[1]]

        if (~block_region).sum().item() > 0:
            mask[slices[0], slices[1]] = True
            return True

        return False
