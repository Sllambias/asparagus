import torch
from torch import nn
from typing import Optional


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, rec_loss_masked_only: bool = True) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.rec_loss_masked_only = rec_loss_masked_only

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rec_loss_masked_only and mask is not None:
            assert mask.dtype == torch.bool, "Mask must be boolean"
            masked = ~mask
            assert masked.any(), "Mask contains no masked voxels"
            return self.loss_fn(pred[masked], target[masked])
        return self.loss_fn(pred, target)
