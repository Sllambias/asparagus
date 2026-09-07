import torch
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    def __init__(self, rec_loss_fn: nn.Module, rec_loss_masked_only: bool = True):
        super().__init__()
        self.rec_loss_fn = rec_loss_fn
        self.rec_loss_masked_only = rec_loss_masked_only

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.rec_loss_masked_only and mask is not None:
            assert mask.dtype == torch.bool, "Mask must be boolean"
            masked = ~mask
            assert masked.any(), "Mask contains no masked voxels"
            return self.rec_loss_fn(pred[masked], target[masked])
        else:
            return self.rec_loss_fn(pred, target)
