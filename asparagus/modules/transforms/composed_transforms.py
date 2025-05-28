from torchvision import transforms
from yucca.modules.data.augmentation.transforms import (
    Torch_CopyImageToLabel,
)
from asparagus.modules.transforms.masking import Torch_Mask
from asparagus.modules.transforms.normalize import Torch_Normalize
from yucca.modules.data.augmentation.transforms.cropping_and_padding import Torch_CropPad


def self_supervised_train_transforms(patch_size):
    tforms = transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_CropPad(patch_size=patch_size),
            Torch_Mask(),
            Torch_CopyImageToLabel(copy=True),
        ]
    )
    return tforms
