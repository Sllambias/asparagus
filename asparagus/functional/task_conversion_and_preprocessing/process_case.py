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


import os
from os.path import join
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.pipeline.task_conversion.utils import should_use_volume
import nibabel as nib


def process_mri_case(path, image_save_path, pkl_save_path, preprocessing_config, dtype=torch.float32):
    if os.path.isfile(image_save_path) and os.path.isfile(pkl_save_path):
        return
    try:
        image = nib.load(path)
        case, image_props = preprocess_case_for_training_without_label(
            images=[image], **asdict(preprocessing_config), strict=False
        )
        os.makedirs(os.path.split(image_save_path)[0], exist_ok=True)
        torch.save(torch.tensor(np.array(case), dtype=dtype), image_save_path)
        save_pickle(image_props, pkl_save_path)
        del case, image_props
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")


def process_dwi_case(path, bvals_path, bvecs_path, image_save_path, pkl_save_path, preprocessing_config, dtype=torch.float32):
    if os.path.isfile(image_save_path) and os.path.isfile(pkl_save_path):
        return
    try:
        image = nib.load(path)
        if len(image.shape) == 4:
            bvals = np.loadtxt(bvals_path)
            bvecs = np.loadtxt(bvecs_path)
            if not os.path.exists(bvals_path) or not os.path.exists(bvecs_path):
                logging.error(f"SKIPPED: Missing bval or bvec for: {path}")
                return
            images, bvals = extract_3ddwi_from_4ddwi(image, bvals, bvecs)
        else:
            images = [image]
            bvals = [""]

        for idx, image in enumerate(images):
            case, image_props = preprocess_case_for_training_without_label(
                images=[image], **asdict(preprocessing_config), strict=False
            )
            os.makedirs(os.path.split(image_save_path)[0], exist_ok=True)
            torch.save(torch.tensor(np.array(case), dtype=dtype), image_save_path.replace(".pt", f"_bval{bvals[idx]}.pt"))
            save_pickle(image_props, pkl_save_path.replace(".pkl", f"_bval{bvals[idx]}.pkl"))
            del case, image_props
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")


def process_pet_case(path, image_save_path, pkl_save_path, preprocessing_config, dtype=torch.float32):
    if os.path.isfile(image_save_path) and os.path.isfile(pkl_save_path):
        return
    try:
        image = nib.load(path)
        if len(image.shape) == 4:
            image = extract_3dpet_from_4dpet(image)
        case, image_props = preprocess_case_for_training_without_label(
            images=[image], **asdict(preprocessing_config), strict=False
        )
        os.makedirs(os.path.split(image_save_path)[0], exist_ok=True)
        torch.save(torch.tensor(np.array(case), dtype=dtype), image_save_path)
        save_pickle(image_props, pkl_save_path)
        del case, image_props
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")


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
        if supposed_to_be_3D and len(image.shape) != 3:
            raise ValueError(f"image is not 3D. Shape: {image.shape}.  ")
        if np.min(image.shape) < 20:
            raise ValueError(f"image is too small. Shape: {image.shape}. ")
        if np.count_nonzero(image) < 1:
            raise ValueError(f"image is all zeros. ")
        if np.isnan(np.sum(image)) == True:
            raise ValueError(f"image contains NaN values. ")


def extract_3dpet_from_4dpet(image):
    assert np.min(image.shape) == image.shape[-1]  # Assert that time dimension is indeed last
    image_arr = np.mean(image.get_fdata(), axis=-1)
    header = image.header.copy()
    header.set_data_shape(image_arr.shape)
    image = nib.Nifti1Image(image_arr, image.affine, header=header)
    return image


def extract_3ddwi_from_4ddwi(image, bvals, bvecs, bval_tolerance=50):
    assert np.min(image.shape) == image.shape[-1]  # Assert that time dimension is indeed last
    old_header = image.header.copy()
    # Group similar b-values
    bval_groups = group_bvalues(bvals, tolerance=bval_tolerance)

    dwis = []
    group_bvals = []
    for group_bval, indices in bval_groups:
        dwi = get_data_for_bval_group(group_bval, indices, bvals, bvecs, image.get_fdata())
        if dwi is not None:  # Only add if we got data (not skipped)
            new_header = old_header.copy()
            new_header.set_data_shape(dwi.shape)
            dwi = nib.Nifti1Image(dwi, image.affine, header=new_header)
            dwis.append(dwi)
            group_bvals.append(str(int(round(group_bval))))
        else:
            logging.error(f"Skipped b-value group: {group_bval} (only {len(indices)} volumes)")
    return dwis, group_bvals


def get_best_basis(bvecs):
    """Finds the indices of the three bvecs closest to X, Y, and Z directions."""
    standard_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X direction  # Y direction  # Z direction
    best_match = []
    for std_vec in standard_basis:
        best_idx = None
        max_cosine = -1
        for i in range(bvecs.shape[1]):
            norm_bvec = np.linalg.norm(bvecs[:, i])
            if norm_bvec == 0:
                continue
            cos_sim = np.dot(bvecs[:, i], std_vec) / norm_bvec  # Normalized dot product
            if cos_sim > max_cosine:
                max_cosine = cos_sim
                best_idx = i
        best_match.append(best_idx)
    assert len(best_match) == 3
    return best_match


def group_bvalues(bvals, tolerance=50):
    """Group similar b-values together. First groups zeros, then remaining values."""
    groups = []

    # Handle zeros first, only keeping first indices
    zero_indices = np.where(bvals == 0)[0]
    if len(zero_indices) > 0:
        # Only use the first indices of zeros
        groups.append((0, zero_indices[:1]))

    # Handle non-zero values
    non_zero_mask = bvals != 0
    non_zero_bvals = bvals[non_zero_mask]

    if len(non_zero_bvals) > 0:
        unique_non_zero = np.unique(non_zero_bvals)
        used_indices = set(zero_indices[:1]) if len(zero_indices) > 0 else set()

        for bval in sorted(unique_non_zero):
            similar_indices = np.where((bvals >= bval - tolerance) & (bvals <= bval + tolerance) & non_zero_mask)[0]

            # Exclude already used indices
            available_indices = [idx for idx in similar_indices if idx not in used_indices]

            if len(available_indices) >= 3:  # Only form groups with at least 3 volumes
                group_representative = np.mean(bvals[available_indices])
                groups.append((group_representative, np.array(available_indices)))

                # Mark these indices as used
                used_indices.update(available_indices)

    return groups


def get_data_for_bval_group(group_bval, indices, bvals, bvecs, data):
    """Extract data for a group of similar b-values."""
    if group_bval == 0:
        # For b0, take the first volume
        return data[..., indices[0]]

    # For non-zero b-values, we need at least 3 volumes
    if len(indices) < 3:
        return None  # Skip this group

    # Get the best basis for this group
    basis_indices = get_best_basis(bvecs[:, indices])
    selected_volumes = data[..., [indices[i] for i in basis_indices]]
    return np.mean(selected_volumes, axis=-1)
