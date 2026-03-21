import torch
import torch.nn as nn
from asparagus.functional.decorators import depends_on_timm
from einops import rearrange
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.eva import Eva
from gardening_tools.modules.networks.components.heads import ClsRegHead
from gardening_tools.modules.networks.components.transformer import PatchEmbed
from gardening_tools.modules.networks.components.weight_init import InitWeights_He
from typing import Tuple

try:
    from gardening_tools.modules.networks.primus import Primus
    from timm.layers import RotaryEmbeddingCat
except ImportError:
    print(
        "Primus could not be imported. To use this network the optional dependency timm must be installed. "
        "timm can be installed manually or with pip install asparagus[extras]"
    )


class PrimusCLSREG(BaseNet):
    """Late-fusion classification/regression network using a Primus backbone.

    Each input modality is processed independently through a shared encoder (PatchEmbed + Eva).
    The resulting features are concatenated along the channel dimension and passed through
    a ClsRegHead (global pool + linear).

    Uses the same attribute names (encoder, eva) as Primus for checkpoint weight compatibility.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        eva_depth: int,
        eva_numheads: int,
        input_shape: Tuple[int, ...],
        dropout_rate: float = 0.0,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
        late_fusion: bool = False,
    ):
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3d is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()

        self.num_classes = output_channels
        self.late_fusion = late_fusion
        self.stem_weight_name = "encoder.proj.weight"

        self.encoder = PatchEmbed(patch_embed_size, 1 if late_fusion else input_channels, embed_dim)
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=0.0,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        # Classification/regression head (replaces PatchDecode)
        self.decoder = ClsRegHead(
            pool_op=nn.AdaptiveAvgPool3d,
            input_channels=embed_dim * input_channels if late_fusion else embed_dim,
            output_channels=output_channels,
            dropout_rate=dropout_rate,
        )

        self.encoder.apply(InitWeights_He(1e-2))

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """Restore the full sequence by filling blanks with mask tokens and reordering."""
        if keep_indices is None:
            return x, None
        B, num_kept, C = x.shape
        device = x.device

        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        restored = torch.zeros(B, num_patches, C, device=device)
        restored_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

        for i in range(B):
            kept_pos = keep_indices[i]
            all_indices = torch.arange(num_patches, device=device)
            mask = torch.ones(num_patches, device=device, dtype=torch.bool)
            mask[kept_pos] = False
            masked_pos = all_indices[mask]

            restored[i, kept_pos] = x[i]
            restored[i, masked_pos] = mask_tokens[i, : len(masked_pos)]
            restored_mask[i, kept_pos] = True

        return (restored, restored_mask)

    def _encode_single(self, x):
        """Encode a single-modality input through PatchEmbed + Eva."""
        x = self.encoder(x)
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (h w d) c")
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1] :]

        restored_x, _ = self.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (h w d) c -> b c w h d", h=H, w=W, d=D)
        return x

    def _encode(self, x):
        if not self.late_fusion:
            return self.encoder(x)  # early-fusion of modalities

        # late-fusion of modalities
        B, N = x.shape[:2]
        x = self._encode_single(x.view(B * N, -1, *x.shape[2:]))
        return x.reshape(B, N * x.shape[1], *x.shape[2:])

    def forward(self, x):
        x = self._encode(x)
        return self.decoder([x])

    def freeze_backbone(self):
        """Freeze the encoder and EVA backbone for linear probing."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        for param in self.eva.parameters():
            param.requires_grad = False
        self.eva.eval()


@depends_on_timm()
def primus_s(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=396,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=12,
        eva_numheads=6,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
    )
    return model


@depends_on_timm()
def primus_b(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=792,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=12,
        eva_numheads=12,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_m(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=864,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=16,
        eva_numheads=12,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_l(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=1056,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=24,
        eva_numheads=16,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_h(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=1248,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=32,
        eva_numheads=16,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_g(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=1584,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=32,
        eva_numheads=24,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_m_clsreg(
    input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), dropout_rate=0.0, late_fusion: bool = False
):
    return PrimusCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        embed_dim=864,
        patch_embed_size=patch_embed_size,
        eva_depth=16,
        eva_numheads=12,
        input_shape=patch_size,
        dropout_rate=dropout_rate,
        scale_attn_inner=True,
        init_values=0.1,
        late_fusion=late_fusion,
    )


if __name__ == "__main__":
    net = primus_s(1, 1, (64, 64, 64))
