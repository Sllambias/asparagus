"""
Spatial and mask-related metrics for SSL pretraining.
"""

import torch
from typing import Any, Dict, Optional


def compute(mask: Optional[torch.Tensor], x: torch.Tensor) -> Dict[str, Any]:
    """
    Masking strategy statistics for MAE-style pretraining.
    High mask_ratio_std indicates non-uniform masking patterns.
    """
    if mask is not None:
        mask_ratio_realized = (1 - mask.float().mean()).item()
        mask_ratio_std = mask.float().std().item()
        visible_tokens = mask.sum().item()
        masked_tokens = (~mask).sum().item()
    else:
        mask_ratio_realized = 0.0
        mask_ratio_std = 0.0
        visible_tokens = 0
        masked_tokens = 0

    return {
        "mask_ratio_realized": mask_ratio_realized,
        "mask_ratio_std": mask_ratio_std,
        "visible_tokens": visible_tokens,
        "masked_tokens": masked_tokens,
    }
