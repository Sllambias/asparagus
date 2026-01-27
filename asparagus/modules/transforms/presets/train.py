from gardening_tools.functional.transforms.spatial import get_max_rotated_size
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CenterCrop, Torch_CropPad, Torch_Pad
from gardening_tools.modules.transforms.deep_supervision import Torch_DownsampleSegForDS
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_CT_NormalizeC0, Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_Resize, Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from torchvision import transforms


def none(ndim=3, deep_supervision=False):
    return None


def CPU_seg_train_transforms(patch_size, normalize=True):
    if len(patch_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = get_max_rotated_size(patch_size)
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_CropPad(patch_size=pre_aug_patch_size, p_oversample_foreground=0.33),
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_crop(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_CT_C0_clsreg_train_transforms_crop(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_CT_NormalizeC0(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_resize(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Resize(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_val_test_transforms_crop(target_size, normalize=True):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


def CPU_CT_C0_clsreg_val_test_transforms_crop(target_size, normalize=True):
    return transforms.Compose(
        [
            Torch_CT_NormalizeC0(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


def CPU_seg_val_transforms(patch_size, normalize=True):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_CropPad(patch_size=patch_size),
        ]
    )


def CPU_seg_test_transforms(normalize=True):
    return transforms.Compose(
        [
            Torch_Normalize(
                normalize=normalize
            ),  # Torch_Pad(patch_size=min_patch_size) # TODO: This does not work with reverse preprocessing...
        ]
    )


def GPU_all_train_transforms(ndim=3, deep_supervision=False):
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.15),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))

    return tforms
