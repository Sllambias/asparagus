from asparagus.modules.losses.ibot import IBOTPatchLoss3D
from lightly.loss import DINOLoss, KoLeoLoss
from lightly.utils.scheduler import linear_warmup_schedule
from torch import nn


class DINOv2Loss(nn.Module):
    def __init__(
        self,
        teacher_temp_warmup_steps: int = 0,
        student_temp: float = 0.1,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        center_momentum: float = 0.9,
        output_dim: int = 65536,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.teacher_temp_warmup_steps = teacher_temp_warmup_steps
        self.center_momentum = center_momentum
        self.output_dim = output_dim
        self.w_ibot = ibot_loss_weight
        self.w_koleo = koleo_loss_weight
        # Initialize loss functions with dynamic temperature
        self.dino_loss_fn = DINOLoss(
            output_dim=self.output_dim,
            teacher_temp=self.teacher_temp_min,  # Will be updated dynamically
            student_temp=self.student_temp,
            center_momentum=self.center_momentum,
        )

        self.ibot_loss_fn = (
            IBOTPatchLoss3D(
                output_dim=self.output_dim,
                teacher_temp=self.teacher_temp_min,  # Will be updated dynamically
                student_temp=self.student_temp,
                center_momentum=self.center_momentum,
            )
            if self.w_ibot > 0
            else None
        )

        self.koleo_loss_fn = KoLeoLoss() if self.w_koleo > 0 else None

    def get_teacher_temperature(self, global_step: int) -> float:
        """Calculate teacher temperature with linear warmup using PyTorch Lightning trainer info."""

        return linear_warmup_schedule(
            step=global_step,
            warmup_steps=self.teacher_temp_warmup_steps,
            start_value=self.teacher_temp_min,
            end_value=self.teacher_temp_max,
        )

    def forward(self, input_dict, batch_size, global_step: int = 0):
        """
        input_dict: {
            "teacher_cls_out": Tensor,
            "student_cls_out": Tensor,
            "masked_teacher_patch_tokens_after_head": Tensor [N_masked, C] (optional),
            "masked_student_global_patch_tokens_after_head": Tensor [N_masked, C] (optional),
            "student_global_cls_token_before_head": Tensor (optional),
            "mask": Tensor (optional, for patch loss),
        }
        Computes the DINOV2 loss for 3D data with dynamic teacher temperature
        """

        teacher_outputs = input_dict.get("teacher_cls_token", None)
        student_outputs = input_dict.get("student_cls_token", None)
        teacher_patches = input_dict.get("masked_teacher_patch_tokens_after_head", None)
        student_patches = input_dict.get("masked_student_global_patch_tokens_after_head", None)
        student_global_cls_features = input_dict.get("student_global_cls_token_before_head", None)
        mask = input_dict.get("mask", None)

        if student_outputs is None or teacher_outputs is None:
            raise ValueError("Both student_outputs and teacher_outputs must be provided.")

        # Calculate dynamic teacher temperature
        teacher_temp = self.get_teacher_temperature(global_step)

        # Update temperature in loss functions
        self.dino_loss_fn.teacher_temp = teacher_temp
        if self.ibot_loss_fn is not None:
            self.ibot_loss_fn.teacher_temp = teacher_temp

        # DINO loss - proper chunking based on views
        dino_loss = self.dino_loss_fn(
            teacher_out=teacher_outputs.chunk(teacher_outputs.size(0) // batch_size),  # 2 global teacher views
            student_out=student_outputs.chunk(student_outputs.size(0) // batch_size),  # All student views
            teacher_temp=teacher_temp,
        )
        # Patch-level iBOT loss
        if self.ibot_loss_fn is not None:
            if teacher_patches is None or student_patches is None or mask is None:
                raise ValueError("iBOT loss requires the masked teacher and student patch tokens and the mask.")

            ibot_loss = self.ibot_loss_fn(
                teacher_out=teacher_patches,
                student_out=student_patches,
                mask=mask,
                teacher_temp=teacher_temp,
            )
        else:
            ibot_loss = 0

        # KoLeo on the backbone CLS tokens of the global views, one view at a time, as in the official implementation
        koleo_loss = (
            sum(self.koleo_loss_fn(s) for s in student_global_cls_features.chunk(2)) if self.koleo_loss_fn is not None else 0
        )

        losses = {
            "dino_loss": dino_loss,
            "ibot_loss": ibot_loss,
            "koleo_loss": koleo_loss,
            "total_loss": dino_loss + self.w_ibot * ibot_loss + self.w_koleo * koleo_loss,
            "teacher_temp": teacher_temp,
        }

        return losses
