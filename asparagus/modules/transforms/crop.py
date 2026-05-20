import numpy as np
import torch
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


def select_foreground_voxel_to_include(foreground_locations):
    if isinstance(foreground_locations, list):
        locidx = np.random.choice(len(foreground_locations))
        location = foreground_locations[locidx]
    elif isinstance(foreground_locations, dict):
        selected_class = np.random.choice(list(foreground_locations.keys()))
        locidx = np.random.choice(len(foreground_locations[selected_class]))
        location = foreground_locations[selected_class][locidx]
    return location


def torch_crop(
    image: torch.tensor,
    patch_size,
    input_dims: torch.tensor,
    target_image_shape: list | tuple,
    target_label_shape: list | tuple,
    p_oversample_foreground=0.0,
    foreground_locations=None,
    label: torch.tensor = None,
):
    if foreground_locations is None:
        foreground_locations = []

    if len(patch_size) == 3:
        image, label = torch_crop_3D_case_from_3D(
            image=image,
            foreground_locations=foreground_locations,
            label=label,
            patch_size=patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
        )
    elif len(patch_size) == 2 and input_dims == 3:
        image, label = torch_crop_2D_case_from_3D(
            image=image,
            foreground_locations=foreground_locations,
            label=label,
            patch_size=patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
        )
    elif len(patch_size) == 2 and input_dims == 2:
        image, label = torch_crop_2D_case_from_2D(
            image=image,
            foreground_locations=foreground_locations,
            label=label,
            patch_size=patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
        )

    return image, label


def torch_crop_3D_case_from_3D(
    image,
    foreground_locations,
    label,
    patch_size,
    p_oversample_foreground,
    target_image_shape,
    target_label_shape,
):
    image_out = torch.zeros(target_image_shape, device=image.device)
    label_out = torch.zeros(target_label_shape, device=image.device)

    crop_start_idx = []
    if len(foreground_locations) == 0 or np.random.uniform() >= p_oversample_foreground:
        for d in range(3):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [np.random.randint(image.shape[d + 1] - patch_size[d] + 1)]
    else:
        location = select_foreground_voxel_to_include(foreground_locations)
        for d in range(3):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [
                    np.random.randint(
                        max(0, location[d] - patch_size[d]),
                        min(location[d], image.shape[d + 1] - patch_size[d]) + 1,
                    )
                ]

    image_out = image[
        :,
        crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
        crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
        crop_start_idx[2] : crop_start_idx[2] + patch_size[2],
    ]
    if label is None:
        return image_out, None
    label_out = label[
        :,
        crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
        crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
        crop_start_idx[2] : crop_start_idx[2] + patch_size[2],
    ]
    return image_out, label_out


def torch_crop_2D_case_from_3D(
    image,
    foreground_locations,
    label,
    patch_size,
    p_oversample_foreground,
    target_image_shape,
    target_label_shape,
):
    image_out = torch.zeros(target_image_shape, device=image.device)
    label_out = torch.zeros(target_label_shape, device=image.device)

    crop_start_idx = []
    if len(foreground_locations) == 0 or np.random.uniform() >= p_oversample_foreground:
        x_idx = np.random.randint(image.shape[1])
        for d in range(2):
            if image.shape[d + 2] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [np.random.randint(image.shape[d + 2] - patch_size[d] + 1)]
    else:
        location = select_foreground_voxel_to_include(foreground_locations)
        x_idx = location[0]
        for d in range(2):
            if image.shape[d + 2] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [
                    np.random.randint(
                        max(0, location[d + 1] - patch_size[d]),
                        min(location[d + 1], image.shape[d + 2] - patch_size[d]) + 1,
                    )
                ]

    image_out[:, :, :] = image[
        :,
        x_idx,
        crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
        crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
    ]

    if label is None:
        return image_out, None

    label_out[:, :, :] = label[
        :,
        x_idx,
        crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
        crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
    ]

    return image_out, label_out


def torch_crop_2D_case_from_2D(
    image,
    foreground_locations,
    label,
    patch_size,
    p_oversample_foreground,
    target_image_shape,
    target_label_shape,
):
    image_out = torch.zeros(target_image_shape, device=image.device)
    label_out = torch.zeros(target_label_shape, device=image.device)

    crop_start_idx = []
    if len(foreground_locations) == 0 or np.random.uniform() >= p_oversample_foreground:
        for d in range(2):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [np.random.randint(image.shape[d + 1] - patch_size[d] + 1)]
    else:
        location = select_foreground_voxel_to_include(foreground_locations)
        for d in range(2):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [
                    np.random.randint(
                        max(0, location[d] - patch_size[d]),
                        min(location[d], image.shape[d + 1] - patch_size[d]) + 1,
                    )
                ]

    image_out[:, :, :] = image[
        :,
        crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
        crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
    ]

    if label is None:
        return image_out, None

    label_out[:, :, :] = label[
        :,
        crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
        crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
    ]

    return image_out, label_out


class Torch_Crop(BaseTransform):
    def __init__(
        self,
        data_key: str = "image",
        label_key: str = "label",
        patch_size: tuple | list = None,
        p_oversample_foreground: float = 0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(data, target_shape):
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape

    def __crop__(
        self,
        data_dict,
        foreground_locations,
        input_shape,
        p_oversample_foreground,
        target_image_shape,
        target_label_shape,
    ):
        image = data_dict[self.data_key]
        label = data_dict.get(self.label_key)
        image, label = torch_crop(
            image=image,
            patch_size=self.patch_size,
            input_dims=len(input_shape[1:]),
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            p_oversample_foreground=p_oversample_foreground,
            foreground_locations=foreground_locations,
            label=label,
        )
        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, data_dict: dict) -> dict:
        input_shape, target_image_shape, target_label_shape = self.get_params(
            data=data_dict[self.data_key],
            target_shape=self.patch_size,
        )
        return self.__crop__(
            data_dict=data_dict,
            foreground_locations=data_dict.get("foreground_locations"),
            input_shape=input_shape,
            p_oversample_foreground=self.p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
        )
