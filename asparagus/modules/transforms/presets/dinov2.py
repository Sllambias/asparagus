import torch
from asparagus.modules.transforms.as_image import Torch_AsImage
from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.sampling import Torch_Resize
from torchvision import transforms
from typing import Any, List, Optional, Sequence, Tuple, Union


def dinov2(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    **kwargs: Any,
) -> DINOv2Augmentation:
    """The official DINOv2 augmentation. The other arguments go to DINOv2Augmentation."""
    return DINOv2Augmentation(global_view_size=global_view_size, local_view_size=local_view_size, **kwargs)


def ultradino(
    global_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    local_view_size: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: Optional[List[int]] = None,  # unused, passed by the pretrain pipeline
    **kwargs: Any,
) -> DINOv2Augmentation:
    """UltraDino's augmentation (DataAugmentationDINUS in Dinus): the official one with vertical flips, no hue jitter,
    no RandomGrayscale and the FUS13M mean and std. The other arguments go to DINOv2Augmentation."""
    return DINOv2Augmentation(
        global_view_size=global_view_size,
        local_view_size=local_view_size,
        vertical_flip=True,
        hue=0.0,
        grayscale_prob=0.0,
        normalize_mean=[0.1842924807],
        normalize_std=[0.2187705424],
        **kwargs,
    )


def ultradino_clsreg_val_test(target_size: Sequence[int]) -> transforms.Compose:
    """8-bit images resized whole to target_size and normalized with the FUS13M mean and std, as UltraDino pretrains."""
    return transforms.Compose(
        [
            Torch_AsImage(transforms.ConvertImageDtype(torch.float32)),
            Torch_Resize(target_size=target_size, clip_to_input_range=True, antialias=True, mode="bicubic"),
            Torch_AsImage(transforms.Normalize(mean=[0.1842924807], std=[0.2187705424])),
        ]
    )


def ultradino_clsreg_train(target_size: Sequence[int]) -> transforms.Compose:
    """ultradino_clsreg_val_test with random flips along every axis and brightness and contrast jitter."""
    return transforms.Compose(
        [
            Torch_AsImage(transforms.ConvertImageDtype(torch.float32)),
            Torch_Resize(target_size=target_size, clip_to_input_range=True, antialias=True, mode="bicubic"),
            Torch_Mirror(p_per_sample=1.0, axes=tuple(range(len(target_size))), p_mirror_per_axis=0.5),
            Torch_AsImage(transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.1)),
            Torch_AsImage(transforms.Normalize(mean=[0.1842924807], std=[0.2187705424])),
        ]
    )


def ultradino_mae_train(patch_size: Sequence[int]) -> transforms.Compose:
    """ultradino_clsreg_train at patch_size, for masked autoencoder pretraining with asp_pretrain."""
    return ultradino_clsreg_train(target_size=patch_size)


def ultradino_mae_val(patch_size: Sequence[int]) -> transforms.Compose:
    """ultradino_clsreg_val_test at patch_size, for masked autoencoder pretraining with asp_pretrain."""
    return ultradino_clsreg_val_test(target_size=patch_size)
