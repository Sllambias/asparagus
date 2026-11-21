"""
FROM: https://github.com/AIM-Harvard/DINOv2-3D-Med/blob/master/models/backbones/masked_vit_wrapper.py
MaskedVisionTransformerMONAI3D: Wrapper for MONAI ViT with 3D masked image modeling support.
Handles dynamic input sizes and patch masking for self-supervised learning.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth
from typing import Optional, Sequence


class LayerScale(nn.Module):
    """Learned per-channel scale of a residual branch, as in the official DINOv2."""

    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class MaskedVisionTransformer(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        layer_scale_init: Optional[float] = None,
        drop_path_rate: float = 0.0,
        num_register_tokens: int = 0,
    ):
        """
        Args:
            vit: MONAI ViT instance
            layer_scale_init: Initial LayerScale value of the residual branches, None for no LayerScale
            drop_path_rate: Probability of dropping a residual branch per sample, the same in every block
            num_register_tokens: Number of register tokens, which are left out of the output
        """
        super().__init__()
        self.vit = vit

        # Extract configuration from ViT
        self.hidden_dim = getattr(vit.patch_embedding.patch_embeddings, "out_channels", 768)
        self.sequence_length = getattr(vit.patch_embedding, "n_patches") + 1  # +1 for cls token
        self.patch_size = getattr(vit.patch_embedding.patch_embeddings, "kernel_size")
        self.spatial_dims = len(self.patch_size)

        self.grid_size = [round(self.sequence_length ** (1 / self.spatial_dims))] * self.spatial_dims

        # MONAI's classification head is unused; freezing it lets DDP run without find_unused_parameters
        if hasattr(self.vit, "classification_head"):
            for param in self.vit.classification_head.parameters():
                param.requires_grad = False

        # Initialized to zeros, as in the official implementation
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # Set on the blocks so the parameters are named and learning rate decayed per block, as official
        for block in self.vit.blocks:
            if layer_scale_init is None:
                block.ls1 = nn.Identity()
                block.ls2 = nn.Identity()
            else:
                block.ls1 = LayerScale(self.hidden_dim, layer_scale_init)
                block.ls2 = LayerScale(self.hidden_dim, layer_scale_init)
        self.drop_path = StochasticDepth(drop_path_rate, mode="row")

        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, self.hidden_dim))
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None
        # Official DINOv2 interpolates bicubically; 3D has no bicubic mode, so it uses trilinear
        if self.spatial_dims == 2:
            self.interpolation_mode = "bicubic"
        elif self.spatial_dims == 3:
            self.interpolation_mode = "trilinear"

    def interpolate_pos_embed(self, grid_size: Sequence[int]) -> torch.Tensor:
        """Position embeddings for a patch grid, interpolated as in the official DINOv2 when the grid size differs."""
        pos_embed = self.vit.patch_embedding.position_embeddings
        if list(grid_size) == list(self.grid_size):
            return pos_embed
        pos_embed = pos_embed.float().transpose(1, 2).reshape(1, self.hidden_dim, *self.grid_size)
        # The official offset of 0.1 keeps floating point error from shrinking the output grid
        scale_factor = [(new_size + 0.1) / size for new_size, size in zip(grid_size, self.grid_size)]
        pos_embed = F.interpolate(pos_embed, scale_factor=scale_factor, mode=self.interpolation_mode, align_corners=False)
        assert list(pos_embed.shape[2:]) == list(grid_size), f"Interpolated to {pos_embed.shape[2:]}, expected {grid_size}"
        return pos_embed.flatten(2).transpose(1, 2).to(self.vit.patch_embedding.position_embeddings.dtype)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional masking and dynamic input sizes.
        Args:
            x: Input tensor (B, C, H, W) or (B, C, D, H, W)
            mask: Optional boolean mask over the patch tokens (B, num_patches), True where a patch is masked
        Returns:
            Normalized tokens (B, 1 + num_patches, hidden_dim), CLS token first
        """
        embeddings = self.vit.patch_embedding.patch_embeddings(x)
        grid_size = embeddings.shape[2:]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        # Mask before adding the position embedding so masked tokens keep their position, as in the official implementation
        if mask is not None:
            embeddings = torch.where(mask.unsqueeze(-1), self.mask_token.to(embeddings.dtype), embeddings)

        embeddings = embeddings + self.interpolate_pos_embed(grid_size)
        cls_tokens = self.vit.cls_token.expand(x.shape[0], -1, -1)
        tokens = [cls_tokens, embeddings]
        # Registers come after the position embedding, so they have no position, as official
        if self.register_tokens is not None:
            tokens.insert(1, self.register_tokens.expand(x.shape[0], -1, -1))
        embeddings = torch.cat(tokens, dim=1)

        # MONAI's TransformerBlock.forward with LayerScale and drop path added to both residual branches
        for blk in self.vit.blocks:
            embeddings = embeddings + self.drop_path(blk.ls1(blk.attn(blk.norm1(embeddings))))
            embeddings = embeddings + self.drop_path(blk.ls2(blk.mlp(blk.norm2(embeddings))))

        embeddings = self.vit.norm(embeddings)
        return torch.cat((embeddings[:, :1], embeddings[:, 1 + self.num_register_tokens :]), dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard forward pass returning embeddings.
        """
        return self.encode(x, mask=mask)
