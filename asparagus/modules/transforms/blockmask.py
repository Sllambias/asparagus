"""
RandomBlockMask: block masking for DINOv2/iBOT in 2D and 3D.
Follows the official DINOv2 masking (dinov2/data/masking.py and collate.py); 3D uses the same scheme with 3D blocks.
"""

import math
import random
import torch
from torch import Tensor, nn
from typing import List, Optional, Tuple, Union


class RandomBlockMask(nn.Module):
    def __init__(
        self,
        grid_size: Union[Tuple[int, int], Tuple[int, int, int]],
        mask_probability: float = 0.5,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
        min_num_patches: int = 4,
        max_num_patches: Optional[float] = None,
        min_aspect: float = 0.3,
    ):
        """
        Args:
            grid_size: Patch grid of one view, (H, W) or (D, H, W).
            mask_probability: Share of the samples in a batch that get masked.
            mask_ratio_min: Lowest share of patches masked in a masked sample.
            mask_ratio_max: Highest share of patches masked in a masked sample.
            min_num_patches: Smallest block size to aim for.
            max_num_patches: Largest block size, half the grid by default.
            min_aspect: Smallest aspect ratio of a block; the largest is its inverse.
        """
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.num_patches = math.prod(self.grid_size)
        self.mask_probability = mask_probability
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.min_num_patches = min_num_patches
        self.max_num_patches = 0.5 * self.num_patches if max_num_patches is None else max_num_patches
        self.log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

    def forward(self, batch_size: int, device: Optional[Union[torch.device, str]] = None) -> Tensor:
        """Boolean masks of shape (batch_size, *grid_size), True where a patch is masked."""
        n_samples_masked = int(batch_size * self.mask_probability)
        ratios = torch.linspace(self.mask_ratio_min, self.mask_ratio_max, n_samples_masked + 1)
        masks = []
        for i in range(n_samples_masked):
            ratio = random.uniform(ratios[i], ratios[i + 1])
            masks.append(self.block_mask(int(self.num_patches * ratio)))
        for _ in range(n_samples_masked, batch_size):
            masks.append(self.block_mask(0))
        random.shuffle(masks)
        return torch.stack(masks).to(device)

    def block_mask(self, num_masking_patches: int) -> Tensor:
        """Mask up to num_masking_patches patches with random blocks."""
        mask = torch.zeros(self.grid_size, dtype=torch.bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = min(num_masking_patches - mask_count, self.max_num_patches)
            delta = self.add_block(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask

    def add_block(self, mask: Tensor, max_mask_patches: float) -> int:
        """Try 10 times to add a block that masks between 1 and max_mask_patches new patches."""
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            sides = self.block_sides(target_area)
            if any(side >= size for side, size in zip(sides, self.grid_size)):
                continue
            starts = [random.randint(0, size - side) for side, size in zip(sides, self.grid_size)]
            block = tuple(slice(start, start + side) for start, side in zip(starts, sides))
            delta = int((~mask[block]).sum())
            if 0 < delta <= max_mask_patches:
                mask[block] = True
                return delta
        return 0

    def block_sides(self, target_area: float) -> List[int]:
        """Side lengths of a block of about target_area patches, with random aspect ratios between the sides."""
        log_offsets = [0.0] * len(self.grid_size)
        for i in range(len(self.grid_size) - 1):
            log_aspect = random.uniform(*self.log_aspect_ratio)
            log_offsets[i] += log_aspect / 2
            log_offsets[i + 1] -= log_aspect / 2
        side = target_area ** (1 / len(self.grid_size))
        return [max(1, round(side * math.exp(offset))) for offset in log_offsets]
