import numpy as np
import nibabel as nib
import os
import logging
import torch
from typing import Union, List, Optional
from yucca.functional.preprocessing import (
    determine_target_size,
    resample_and_normalize_case,
    pad_case_to_size,
    apply_nifti_preprocessing_and_return_numpy,
)
from dataclasses import asdict
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from yucca.functional.array_operations.transpose import transpose_case
from yucca.functional.array_operations.bounding_boxes import get_bbox_for_foreground
from yucca.functional.array_operations.cropping_and_padding import crop_to_box


def process_mri_case(path, image_save_path, pkl_save_path, preprocessing_config, dtype=torch.float32):
    if os.path.isfile(image_save_path) and os.path.isfile(pkl_save_path):
        return
    try:
        image = nib.load(path)
        case, image_props = preprocess_case_for_training_without_label(
            images=[image], **asdict(preprocessing_config), strict=False
        )
        os.makedirs(os.path.split(image_save_path)[0], exist_ok=True)
        torch.save(torch.tensor(case, dtype=dtype), image_save_path)
        save_pickle(image_props, pkl_save_path)
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")


def process_dwi_case(path, root, output_dir, dtype=np.float16):
    raise NotImplementedError("DWI case processing is not implemented yet.")


def process_pet_case(path, root, output_dir, dtype=np.float16):
    raise NotImplementedError("PET case processing is not implemented yet.")


def preprocess_case_for_training_without_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    normalization_operation: list,
    background_pixel_value: int = 0,
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    transpose: Optional[list] = [0, 1, 2],
    strict: bool = True,
    supposed_to_be_3D: bool = True,
):
    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=None,
        include_header=False,
        strict=strict,
    )

    images = [image.squeeze() for image in images]
    logging.error(images)
    verify_3D_image_is_valid(images, supposed_to_be_3D=supposed_to_be_3D)
    original_size = images[0].shape

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        image_properties["crop_to_nonzero"] = nonzero_box
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
    else:
        image_properties["crop_to_nonzero"] = crop_to_nonzero

    image_properties["size_before_transpose"] = list(images[0].shape)

    images = transpose_case(images, axes=transpose)

    image_properties["size_after_transpose"] = list(images[0].shape)

    resample_target_size, final_target_size, new_spacing = determine_target_size(
        images_transposed=images,
        original_spacing=np.array(image_properties["nifti_metadata"]["original_spacing"]),
        transpose_forward=transpose,
        target_size=target_size,
        target_spacing=target_spacing,
        keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
    )

    images = resample_and_normalize_case(
        case=images,
        target_size=resample_target_size,
        norm_op=normalization_operation,
        intensities=intensities,
    )

    if final_target_size is not None:
        images = pad_case_to_size(case=images, size=final_target_size, label=None)

    image_properties["new_size"] = list(images[0].shape)
    image_properties["foreground_locations"] = []
    image_properties["original_spacing"] = image_properties["nifti_metadata"]["original_spacing"]
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    return images, image_properties


def verify_3D_image_is_valid(images: list, supposed_to_be_3D: bool = True):
    for image in images:
        print(image)
        print(images)
        if supposed_to_be_3D and len(image.shape) != 3:
            raise ValueError(f"image is not 3D. Shape: {image.shape}.  ")
        if np.min(image.shape) < 20:
            raise ValueError(f"image is too small. Shape: {image.shape}. ")
        if np.count_nonzero(image) < 1:
            raise ValueError(f"image is all zeros. ")
        print(np.isnan(np.sum(image)))
        if np.isnan(np.sum(image)) == True:
            raise ValueError(f"image contains NaN values. ")
