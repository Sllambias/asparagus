import torch
import torch.nn.functional as F
from einops import rearrange


def interpolate_patch_embed_3d(patch_embed, in_shape, out_shape):
    """Resizes patch embeddings using 3D trilinear interpolation.

    Copied from SSL3D_classification/models/eva_mae_openneuro.py
    """
    patch_embed = patch_embed.permute(0, 2, 1)
    patch_embed = rearrange(patch_embed, "B C (x y z) -> B C x y z", **in_shape)
    patch_embed = F.interpolate(patch_embed, size=list(out_shape.values()), mode="trilinear", align_corners=False)
    patch_embed = rearrange(patch_embed, "B C x y z -> B C (x y z)", **out_shape)
    return patch_embed.permute(0, 2, 1)


def resize_pos_embed_3d(
    ckpt_pos_embed, model_pos_embed, num_prefix_tokens, pretrained_target_size, target_size, patch_embed_size
):
    """Resize a pos_embed tensor to match the model's expected shape.

    Separates prefix tokens (cls/register), applies 3D trilinear interpolation
    to the patch tokens, and reattaches the prefix.
    """
    if num_prefix_tokens > 0:
        prefix = ckpt_pos_embed[:, :num_prefix_tokens, :]
        patch_pos_embed = ckpt_pos_embed[:, num_prefix_tokens:, :]
    else:
        prefix = None
        patch_pos_embed = ckpt_pos_embed

    in_shape = {
        "x": pretrained_target_size[0] // patch_embed_size[0],
        "y": pretrained_target_size[1] // patch_embed_size[1],
        "z": pretrained_target_size[2] // patch_embed_size[2],
    }

    out_shape = {
        "x": target_size[0] // patch_embed_size[0],
        "y": target_size[1] // patch_embed_size[1],
        "z": target_size[2] // patch_embed_size[2],
    }

    orig_dtype = patch_pos_embed.dtype
    resized = interpolate_patch_embed_3d(patch_pos_embed.float(), in_shape, out_shape).to(orig_dtype)

    if prefix is not None:
        return torch.cat([prefix, resized], dim=1)
    return resized
