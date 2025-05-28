import torch
from typing import Union
from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
import logging


def volume_wise_znorm(array: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor using volume-wise z-normalization.
    Assumes the input is a 3D tensor (X, Y, Z) or 2D Tensor (X, Y).
    """
    std, mean = torch.std_mean(array)

    return (array - mean) / std


class Torch_Normalize(YuccaTransform):
    def __init__(self, normalize: bool = False, data_key: str = "image", fn=volume_wise_znorm):
        self.normalize = normalize
        self.data_key = data_key
        self.fn = fn

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __normalize__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if isinstance(self.fn, list):
                fn = self.fn[c]
            else:
                fn = self.fn
            data_dict[self.data_key][c] = fn(data_dict[self.data_key][c])
        return data_dict

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        if self.normalize:
            data_dict = self.__normalize__(data_dict)
        return data_dict
