from typing import Type
from torch import nn
from torch.nn.modules.conv import _ConvNd


def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")
