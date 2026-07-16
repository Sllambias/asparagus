"""
FROM: https://github.com/AIM-Harvard/DINOv2-3D-Med/blob/master/models/backbones/masked_vit_wrapper.py
MaskedVisionTransformerMONAI3D: Wrapper for MONAI ViT with 3D masked image modeling support.
Handles dynamic input sizes and patch masking for self-supervised learning.
"""

import torch
from torch import nn


class MaskedVisionTransformer(nn.Module):
    def __init__(self, vit):
        """
        Args:
            vit: MONAI ViT instance
        """
        super().__init__()
        self.vit = vit

        # Extract configuration from ViT
        self.hidden_dim = getattr(vit.patch_embedding.patch_embeddings, "out_channels", 768)
        self.sequence_length = getattr(vit.patch_embedding, "n_patches") + 1  # +1 for cls token
        self.patch_size = getattr(vit.patch_embedding.patch_embeddings, "kernel_size")
        self.spatial_dims = len(self.patch_size)

        self.grid_size = [round(self.sequence_length ** (1 / self.spatial_dims))] * self.spatial_dims

        # Initialize learnable parameters
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        if self.spatial_dims == 2:
            self.interpolation_mode = "bilinear"
        elif self.spatial_dims == 3:
            self.interpolation_mode = "trilinear"

    def encode(self, x, mask=None):
        """
        Forward pass with optional masking and dynamic input sizes.
        Args:
            x: Input tensor (B, C, D, H, W)
            mask: Optional mask tensor (B, sequence_length)
        Returns:
            Patch embeddings (with optional masking)
        """
        B = x.shape[0]

        # Get patch embeddings and reshape if needed
        embeddings = self.vit.patch_embedding.patch_embeddings(x)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        if embeddings.shape[1] != self.vit.patch_embedding.position_embeddings.shape[1]:
            root_n_positions = round(embeddings.shape[1] ** (1 / self.spatial_dims))
            new_grid_size = [root_n_positions] * self.spatial_dims
            x = self.vit.patch_embedding.position_embeddings.transpose(1, 2).reshape(1, self.hidden_dim, *self.grid_size)
            new_pos_embed = (
                torch.nn.functional.interpolate(
                    x,
                    size=new_grid_size,
                    mode=self.interpolation_mode,
                    align_corners=False,
                )
                .reshape(1, self.hidden_dim, -1)
                .transpose(1, 2)
            )
            embeddings = embeddings + nn.Parameter(new_pos_embed)
        else:
            embeddings = embeddings + self.vit.patch_embedding.position_embeddings

        # Add cls token
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Apply masking if provided
        if mask is not None:
            actual_sequence_length = embeddings.shape[1]

            # Adjust mask size if needed
            if mask.shape[1] != actual_sequence_length:
                if mask.shape[1] > actual_sequence_length:
                    mask = mask[:, :actual_sequence_length]
                else:
                    extended_mask = torch.zeros(
                        (B, actual_sequence_length),
                        dtype=torch.bool,
                        device=mask.device,
                    )
                    extended_mask[:, : mask.shape[1]] = mask
                    mask = extended_mask

            # Apply mask tokens
            mask_tokens = self.mask_token.expand(B, actual_sequence_length - 1, -1)
            w = mask[:, 1:].unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings.clone()
            embeddings[:, 1:] = embeddings[:, 1:] * (1 - w) + mask_tokens * w

        # Pass through transformer
        for blk in self.vit.blocks:
            embeddings = blk(embeddings)

        return self.vit.norm(embeddings)

    def forward(self, x, mask=None):
        """
        Standard forward pass returning embeddings.
        """
        return self.encode(x, mask=mask)
