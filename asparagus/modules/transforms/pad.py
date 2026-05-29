import numpy as np
import torch
import torch.nn.functional as F
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from typing import Literal, Union


def get_max_of_image_and_patch(patch_size, image_size):
    if len(patch_size) == 2 and len(image_size) == 3:
        return np.max([patch_size, image_size[1:]], axis=0)
    else:
        return np.max([patch_size, image_size], axis=0)


def torch_pad(
    image: torch.tensor,
    patch_size,
    input_dims: torch.tensor,
    target_image_shape: list | tuple,
    target_label_shape: list | tuple,
    label: torch.tensor = None,
    **pad_kwargs,
):
    if len(patch_size) == 3:
        image, label, pad_box = torch_pad_3D_case_from_3D(
            image=image,
            label=label,
            patch_size=patch_size,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
    elif len(patch_size) == 2 and input_dims == 3:
        image, label, pad_box = torch_pad_2D_case_from_3D(
            image=image,
            label=label,
            patch_size=patch_size,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
    elif len(patch_size) == 2 and input_dims == 2:
        image, label, pad_box = torch_pad_2D_case_from_2D(
            image=image,
            label=label,
            patch_size=patch_size,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )

    return image, label, pad_box


def torch_pad_3D_case_from_3D(
    image,
    label,
    patch_size,
    target_image_shape,
    target_label_shape,
    **pad_kwargs,
):
    image_out = torch.zeros(target_image_shape, device=image.device)
    label_out = torch.zeros(target_label_shape, device=image.device)

    to_pad = []
    for d in range(3):
        if image.shape[d + 1] < patch_size[d]:
            to_pad += [patch_size[d] - image.shape[d + 1]]
        else:
            to_pad += [0]

    pad_lb_x = to_pad[0] // 2
    pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
    pad_lb_y = to_pad[1] // 2
    pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2
    pad_lb_z = to_pad[2] // 2
    pad_ub_z = to_pad[2] // 2 + to_pad[2] % 2

    pad_dict = {"pad_box": [pad_lb_x, pad_ub_x, pad_lb_y, pad_ub_y, pad_lb_z, pad_ub_z], "shape_before_pad": image.shape[1:]}

    image_out = F.pad(
        image,
        (
            pad_lb_z,
            pad_ub_z,
            pad_lb_y,
            pad_ub_y,
            pad_lb_x,
            pad_ub_x,
            0,
            0,
        ),
        **pad_kwargs,
    )
    if label is None:
        return (
            image_out,
            None,
            pad_dict,
        )
    label_out = F.pad(
        label,
        (
            pad_lb_z,
            pad_ub_z,
            pad_lb_y,
            pad_ub_y,
            pad_lb_x,
            pad_ub_x,
            0,
            0,
        ),
    )
    return (
        image_out,
        label_out,
        pad_dict,
    )


def torch_pad_2D_case_from_3D(
    image,
    label,
    patch_size,
    target_image_shape,
    target_label_shape,
    **pad_kwargs,
):
    image_out = torch.zeros(target_image_shape, device=image.device)
    label_out = torch.zeros(target_label_shape, device=image.device)

    # First we pad to ensure min size is met
    to_pad = []
    for d in range(2):
        if image.shape[d + 2] < patch_size[d]:
            to_pad += [patch_size[d] - image.shape[d + 2]]
        else:
            to_pad += [0]

    pad_lb_x = 0
    pad_ub_x = 0
    pad_lb_y = to_pad[0] // 2
    pad_ub_y = to_pad[0] // 2 + to_pad[0] % 2
    pad_lb_z = to_pad[1] // 2
    pad_ub_z = to_pad[1] // 2 + to_pad[1] % 2

    pad_dict = {"pad_box": [pad_lb_x, pad_ub_x, pad_lb_y, pad_ub_y, pad_lb_z, pad_ub_z], "shape_before_pad": image.shape[1:]}

    image_out = F.pad(
        image,
        (
            pad_lb_z,
            pad_ub_z,
            pad_lb_y,
            pad_ub_y,
            pad_lb_x,
            pad_ub_x,
            0,
            0,
        ),
        **pad_kwargs,
    )

    if label is None:
        return image_out, None, pad_dict

    label_out = F.pad(
        label,
        (
            pad_lb_z,
            pad_ub_z,
            pad_lb_y,
            pad_ub_y,
            pad_lb_x,
            pad_ub_x,
            0,
            0,
        ),
    )

    return (
        image_out,
        label_out,
        [
            pad_lb_x,
            pad_ub_x,
            pad_lb_y,
            pad_ub_y,
            pad_lb_z,
            pad_ub_z,
        ],
        pad_dict,
    )


def torch_pad_2D_case_from_2D(
    image,
    label,
    patch_size,
    target_image_shape,
    target_label_shape,
    **pad_kwargs,
):
    image_out = torch.zeros(target_image_shape, device=image.device)
    label_out = torch.zeros(target_label_shape, device=image.device)

    to_pad = []
    for d in range(2):
        if image.shape[d + 1] < patch_size[d]:
            to_pad += [patch_size[d] - image.shape[d + 1]]
        else:
            to_pad += [0]

    pad_lb_x = to_pad[0] // 2
    pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
    pad_lb_y = to_pad[1] // 2
    pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2

    pad_dict = {"pad_box": [pad_lb_x, pad_ub_x, pad_lb_y, pad_ub_y], "shape_before_pad": image.shape[1:]}

    image_out = F.pad(
        image,
        (
            pad_lb_y,
            pad_ub_y,
            pad_lb_x,
            pad_ub_x,
            0,
            0,
        ),
        **pad_kwargs,
    )

    if label is None:
        return image_out, None, pad_dict

    label_out = F.pad(
        label,
        (
            pad_lb_y,
            pad_ub_y,
            pad_lb_x,
            pad_ub_x,
            0,
            0,
        ),
    )

    return image_out, label_out, pad_dict


class Torch_Pad(BaseTransform):
    def __init__(
        self,
        data_key: str = "image",
        label_key: str = "label",
        patch_size: tuple | list = None,
        pad_value: Union[Literal["min", "zero", "replicate"], int, float] = "min",
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.pad_value = pad_value

    @staticmethod
    def get_params(data, pad_value, target_shape):
        if pad_value == "min":
            pad_kwargs = {"value": data.min(), "mode": "constant"}
        elif pad_value == "zero":
            pad_kwargs = {"value": torch.zeros(1, dtype=data.dtype), "mode": "constant"}
        elif isinstance(pad_value, (int, float)):
            pad_kwargs = {"value": pad_value, "mode": "constant"}
        elif pad_value == "replicate":
            pad_kwargs = {"mode": "replicate"}
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape, pad_kwargs

    def __call__(self, data_dict: dict) -> dict:
        image_shape = data_dict[self.data_key].shape
        max_dims = get_max_of_image_and_patch(self.patch_size, image_shape[1:])
        input_shape, target_image_shape, target_label_shape, pad_kwargs = self.get_params(
            data=data_dict[self.data_key],
            pad_value=self.pad_value,
            target_shape=max_dims,
        )
        result = self.__pad__(
            data_dict=data_dict,
            input_shape=input_shape,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
        return result

    def __pad__(
        self,
        data_dict,
        input_shape,
        target_image_shape,
        target_label_shape,
        **pad_kwargs,
    ):
        image = data_dict[self.data_key]
        label = data_dict.get(self.label_key)
        image, label, pad_dict = torch_pad(
            image=image,
            patch_size=self.patch_size,
            input_dims=len(input_shape[1:]),
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            label=label,
            **pad_kwargs,
        )
        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        if data_dict.get("properties") is not None:
            data_dict["properties"].update(pad_dict)
        return data_dict
