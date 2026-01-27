from gardening_tools.functional.transforms.spatial import get_max_rotated_size
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CropPad
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.masking import Torch_Mask
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from torchvision import transforms


def CPU_val_transforms(patch_size):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_CropPad(patch_size=patch_size, p_oversample_foreground=0.4),
            Torch_CopyImageToLabel(copy=True),
        ]
    )


def CPU_train_transforms(patch_size):
    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = get_max_rotated_size(patch_size)
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_CropPad(patch_size=pre_aug_patch_size, p_oversample_foreground=0.4),
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                p_scale_all_channel=p_scale_all_channel,
                clip_to_input_range=False,
                skip_label=False,
            ),
            Torch_CopyImageToLabel(copy=True),
        ]
    )


def GPU_train_transforms(masking=False, ndim=3, mask_ratio=0.6):
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.1),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.2),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.1, p_per_axis=0.3),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if masking:
        tforms.transforms.append(Torch_Mask(ratio=mask_ratio))

    return tforms


def GPU_val_transforms(masking=False, mask_ratio=0.6):
    if masking:
        return transforms.Compose(
            [
                Torch_Mask(ratio=mask_ratio),
            ]
        )
    return None
