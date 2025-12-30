"""
Visualization generation for SSL pretraining logging.
"""

import torch
from typing import Optional


def create_visualizations(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor], epoch: int):
    """
    Create visualization tensors for logging.

    Args:
        x: Input tensor
        y: Ground truth tensor
        pred: Prediction tensor
        mask: Optional mask tensor
        epoch: Current epoch number

    Returns:
        Tuple of (images, error_images) for logging
    """
    from asparagus.functional.visualization import get_logger_compatible_imgs

    # Generate images in a logger-compatible format
    images = get_logger_compatible_imgs(x, y, pred, slice_dim=1, n=1, desc=f"Epoch {epoch}")

    # Also create error maps
    with torch.no_grad():
        error = (pred - y).abs()
        if mask is not None:
            mask_viz = mask.float()
            # Ensure mask has same shape as error for visualization
            if len(mask_viz.shape) < len(error.shape):
                mask_viz = mask_viz.unsqueeze(1)  # Add channel dim if needed
        else:
            mask_viz = torch.ones_like(error)
        error_images = get_logger_compatible_imgs(
            x, error, mask_viz, slice_dim=1, n=1, desc=f"Error Map Epoch {epoch}", titles=["input", "abs error", "mask"]
        )

    return images, error_images
