import torchvision
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Literal
from batchgenerators.utilities.file_and_folder_operations import load_json
from yucca.modules.data.augmentation.transforms.cropping_and_padding import (
    CropPad,
    Torch_CropPad,
)
import logging


class PretrainDataset(Dataset):
    def __init__(
        self,
        files: list,
        patch_size: Tuple[int, int, int],
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.composed_transforms = composed_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data,
        }
        logging.debug(f"Loaded data from {file} with shape {data.shape}")
        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return data_dict
