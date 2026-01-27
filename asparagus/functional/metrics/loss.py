"""
Loss and reconstruction metrics for SSL pretraining.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional


# Frequency-based compute functions
def compute_train(
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> Dict[str, Any]:
    return compute_loss_metrics(loss, pred, y, mask, loss_fn)


def compute_val(
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> Dict[str, Any]:
    return compute_loss_metrics(loss, pred, y, mask, loss_fn) | compute_psnr_metrics(pred, y, mask)


def compute_loss_metrics(
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> Dict[str, Any]:
    return {
        "loss": loss.item(),
        "loss_masked": loss_fn(pred, y, mask).item(),
        "loss_full": loss_fn(pred, y, None).item(),
    }


def compute_psnr_metrics(pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Peak Signal-to-Noise Ratio for reconstruction quality assessment.
    Higher PSNR indicates better perceptual quality (20-40 dB typical for SSL).
    """
    with torch.no_grad():
        mse_full = ((pred - y) ** 2).mean()
        psnr_full = 20 * torch.log10(1.0 / torch.sqrt(mse_full)) if mse_full > 0 else torch.tensor(100.0)

        if mask is not None:
            # PSNR for masked region only
            masked_pred = pred[~mask]
            masked_y = y[~mask]
            if masked_pred.numel() > 0:
                mse_masked = ((masked_pred - masked_y) ** 2).mean()
                psnr_masked = 20 * torch.log10(1.0 / torch.sqrt(mse_masked)) if mse_masked > 0 else torch.tensor(100.0)
            else:
                psnr_masked = torch.tensor(0.0)

            # Unmasked MSE (sanity check - should be close to 0)
            unmasked_pred = pred[mask]
            unmasked_y = y[mask]
            if unmasked_pred.numel() > 0:
                unmasked_mse = ((unmasked_pred - unmasked_y) ** 2).mean()
            else:
                unmasked_mse = torch.tensor(0.0)
        else:
            psnr_masked = psnr_full
            unmasked_mse = torch.tensor(0.0)

        return {"psnr_masked": psnr_masked, "psnr_full": psnr_full, "unmasked_mse": unmasked_mse}
