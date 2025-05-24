import torchvision
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Literal
from batchgenerators.utilities.file_and_folder_operations import load_json
from yucca.modules.data.augmentation.transforms.cropping_and_padding import (
    CropPad,
    Torch_CropPad,
)


class TrainDataset(Dataset):
    def __init__(
        self,
        files: list,
        patch_size: Tuple[int, int, int],
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.croppad = Torch_CropPad(patch_size=self.patch_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data[:-1],
            "label": data[-1:],
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        data_dict = self.croppad(data_dict)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return data_dict
