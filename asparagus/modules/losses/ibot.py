import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightly.models.modules.center import Center
from torch import Tensor, nn


class IBOTPatchLoss3D(nn.Module):
    """
    Patch-level iBOT loss on the masked patch tokens, with 2D or 3D masks.

    Args:
        output_dim: Dimension of the projection head output.
        teacher_temp: Temperature for teacher logits.
        student_temp: Temperature for student logits.
        center_mode: Mode for center calculation (kept for parity with upstream).
        center_momentum: Momentum term for center updates.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """
        Args:
            teacher_out: Teacher logits of the masked patches [N_masked, C].
            student_out: Student logits of the masked patches [N_masked, C].
            mask: Boolean patch mask, shape [B, P] or [B, D, H, W]. True indicates masked tokens.
            teacher_temp: Optional override for teacher temperature.
        Returns:
            Scalar loss.
        """
        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp,
            device=teacher_out.device,
            dtype=teacher_out.dtype,
        )
        mask = mask.flatten(start_dim=1).to(torch.bool)

        if teacher_out.shape != student_out.shape:
            raise ValueError(f"Teacher/student patch shapes differ: {teacher_out.shape} vs {student_out.shape}")
        if teacher_out.shape[0] != mask.sum():
            raise ValueError(f"Got {teacher_out.shape[0]} patch tokens for {mask.sum()} masked patches")

        # Cross-entropy between centered teacher and student distributions
        teacher_softmax = F.softmax((teacher_out - self.center.value.to(teacher_out.device)) / teacher_temperature, dim=-1)
        student_log_softmax = F.log_softmax(student_out / self.student_temp, dim=-1)

        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Per-image weighting as in the reference implementation
        num_masked_per_image = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]
        loss = (loss * weight).sum() / mask.shape[0]

        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """Move the center towards the mean masked teacher token of all devices."""
        total = teacher_out.float().sum(dim=0, keepdim=True)
        count = torch.tensor(float(teacher_out.shape[0]), device=total.device)
        # Every device joins the all-reduce, also with no masked patches, or the others wait for it forever
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total)
            dist.all_reduce(count)
        if count > 0:
            momentum = self.center.momentum
            self.center.center = self.center.center * momentum + total / count * (1 - momentum)
