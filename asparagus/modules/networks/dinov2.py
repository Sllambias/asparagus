import math
import torch
from asparagus.modules.networks.vision_transformer import MaskedVisionTransformer
from functools import partial
from lightly.models.modules.heads import DINOv2ProjectionHead
from lightly.utils.scheduler import cosine_schedule
from monai.networks.nets.vit import ViT
from torch import nn
from typing import Any, Dict, List, Optional, Tuple


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    # One in-place call for all parameters: ema = m * ema + (1 - m) * model
    ema_params = list(model_ema.parameters())
    params = list(model.parameters())
    torch._foreach_lerp_(ema_params, params, 1.0 - m)


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2(nn.Module):
    """
    2D/3D DINOv2 model with teacher/student ViT backbones and DINO/iBOT heads.
    The block masks come from DINOv2Collate in the data loader.
    """

    def __init__(
        self,
        input_channels: int = 1,
        img_size: tuple = (64, 64, 64),
        patch_size: tuple = (16, 16, 16),
        hidden_size: int = 768,
        ibot_separate_head: bool = False,
        projection_dim: int = 65536,
        num_layers: int = 8,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        qkv_bias: bool = False,
        layer_scale_init: Optional[float] = None,
        drop_path_rate: float = 0.0,
        num_register_tokens: int = 0,
    ):
        """
        Initialize DINOv2_3D model.
        Args: see model config for details. Official DINOv2 uses qkv_bias=True, layer_scale_init=1e-5 and
            drop_path_rate=0.3. Drop path is only applied to the student, as official.
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.ibot_separate_head = ibot_separate_head

        self.hidden_size = hidden_size
        vit = partial(
            ViT,
            in_channels=input_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type="conv",
            classification=True,
            spatial_dims=len(img_size),
            qkv_bias=qkv_bias,
        )
        backbone = partial(
            MaskedVisionTransformer,
            layer_scale_init=layer_scale_init,
            num_register_tokens=num_register_tokens,
        )

        self.student_backbone = backbone(vit=vit(), drop_path_rate=drop_path_rate)
        self.student_dino_head = DINOv2ProjectionHead(input_dim=hidden_size, output_dim=projection_dim)
        if ibot_separate_head:
            self.student_ibot_head = DINOv2ProjectionHead(input_dim=hidden_size, output_dim=projection_dim)
        else:
            self.student_ibot_head = self.student_dino_head

        self.teacher_backbone = backbone(vit=vit())
        self.teacher_dino_head = DINOv2ProjectionHead(input_dim=hidden_size, output_dim=projection_dim)
        if ibot_separate_head:
            self.teacher_ibot_head = DINOv2ProjectionHead(input_dim=hidden_size, output_dim=projection_dim)
        else:
            self.teacher_ibot_head = self.teacher_dino_head

        # The teacher starts as a copy of the student, as in the official implementation
        for student, teacher in self.student_teacher_pairs():
            teacher.load_state_dict(student.state_dict())
            freeze_eval_module(teacher)

        grid_size = self.student_backbone.grid_size
        assert math.prod(grid_size) == self.student_backbone.sequence_length - 1, (
            f"Grid size {grid_size} does not match {self.student_backbone.sequence_length - 1} patches"
        )

    def forward_teacher(self, x):
        """Forward pass for the EMA (teacher) backbone without masking."""
        features = self.teacher_backbone(x)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:] if features.shape[1] > 1 else features
        return cls_token, patch_tokens

    def forward_student(self, x, mask=None):
        """Forward pass for the student backbone, keeping patch grid structure."""
        features = self.student_backbone(x, mask=mask)
        cls_tokens = features[:, 0]
        patch_tokens = features[:, 1:] if features.shape[1] > 1 else features
        return cls_tokens, patch_tokens

    def update_teacher(self, global_step: int, max_steps: int) -> None:
        """Update teacher using EMA with cosine momentum schedule."""
        momentum = cosine_schedule(step=global_step, max_steps=max_steps, start_value=0.992, end_value=1.0)
        for student, teacher in self.student_teacher_pairs():
            update_momentum(student, teacher, m=momentum)

    def student_teacher_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        """The student submodules and the teacher submodules that follow them."""
        pairs = [
            (self.student_backbone, self.teacher_backbone),
            (self.student_dino_head, self.teacher_dino_head),
        ]
        if self.ibot_separate_head:
            pairs.append((self.student_ibot_head, self.teacher_ibot_head))
        return pairs

    def forward(
        self,
        global_views: List[torch.Tensor],
        local_views: Optional[List[torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
        ibot: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass for DINOv2 with block masking and multi-view augmentations.
        Args:
            global_views: List of augmented tensors (global views)
            local_views: List of augmented tensors (local views)
            masks: Boolean patch masks of the concatenated global views (2 * B, num_patches), from DINOv2Collate
            ibot: Whether to project the masked patch tokens for the iBOT loss
        Returns:
            Dict of teacher/student outputs for loss computation
        """
        if ibot and masks is None:
            raise ValueError("iBOT needs the masks of the global views. Build the data loader with DINOv2Collate.")
        global_views = torch.cat(global_views)

        if local_views is not None and len(local_views) > 0:
            local_views = torch.cat(local_views)

        with torch.no_grad():
            teacher_cls_token, teacher_patch_tokens = self.forward_teacher(global_views)
            teacher_cls_token = self.teacher_dino_head(teacher_cls_token)

        # Student forward
        student_global_cls_features, student_global_patch_tokens = self.forward_student(global_views, mask=masks)
        student_global_cls_token = self.student_dino_head(student_global_cls_features)

        # Local views
        if local_views is not None:
            student_local_cls_token, _ = self.forward_student(local_views, mask=None)
            student_local_cls_token = self.student_dino_head(student_local_cls_token)
            student_cls_token = torch.cat([student_global_cls_token, student_local_cls_token], dim=0)
        else:
            student_cls_token = student_global_cls_token

        out = {
            "teacher_cls_token": teacher_cls_token,
            "student_cls_token": student_cls_token,
            "student_global_cls_token_before_head": student_global_cls_features,
            "mask": masks,
        }

        # Only the masked patch tokens go through the heads, as in the official implementation
        if ibot:
            with torch.no_grad():
                out["masked_teacher_patch_tokens_after_head"] = self.teacher_ibot_head(teacher_patch_tokens[masks])
            out["masked_student_global_patch_tokens_after_head"] = self.student_ibot_head(student_global_patch_tokens[masks])

        return {"pred": out}

    def encode(self, x: torch.Tensor):
        """
        Simple encoding method that returns the raw CLS token features.
        Useful for feature extraction or as input to downstream tasks.
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            CLS token features of shape (B, hidden_size)
        """
        backbone_features = self.student_backbone(x, mask=None)
        return backbone_features[:, 0]  # Return CLS token only


def vit_test(input_channels=1, img_size=(1, 1, 1), patch_size=(0, 0, 0), hidden_size=120):
    return DINOv2(
        input_channels=input_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        ibot_separate_head=False,
        projection_dim=1024,
    )


def vit_x(
    img_size: tuple = (92, 92, 92),
    patch_size: tuple = (8, 8, 8),
    hidden_size: int = 192,
    **kwargs: Any,
) -> DINOv2:
    """DINOv2 with smaller default sizes. The other arguments go to DINOv2."""
    return DINOv2(img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, **kwargs)
