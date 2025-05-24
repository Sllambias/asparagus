import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from asparagus.functional.task_conversion_and_preprocessing import (
    generate_dataset_json,
    process_mri_case,
    process_dwi_case,
    process_pet_case,
    detect_cases,
    detect_final_cases,
    get_image_and_metadata_output_paths,
    get_bvals_and_bvecs_v1,
    multiprocess_mri_dwi_pet_cases,
    postprocess_standard_dataset,
)
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.presets.preprocessing_presets import GBrainPreprocessingConfig
from itertools import repeat
from multiprocessing.pool import Pool


def convert(path: str = get_source_path(), subdir: str = "ICBM/ICBM_NIFTI", processes=12):
    task_name = "Task027_ICBM"
    file_suffix1 = ".nii.gz"  # e.g. ".nii.gz" or ".nii"
    file_suffix2 = ".mnc"  # e.g. ".nii.gz" or ".nii"
    exclusion_patterns = ["localizer", "Scout", "_Loc"]  # e.g. "func" or "fmri"
    DWI_patterns = ["dti", "diff"]  # e.g. "DWI" or "dwi"
    PET_patterns = []  # e.g. "PET" or "pet"

    source_dir = join(path, subdir)
    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    files_standard1, files_DWI1, files_PET1, files_excluded1 = detect_cases(
        source_dir,
        extension=file_suffix1,
        DWI_patterns=DWI_patterns,
        PET_patterns=PET_patterns,
        exclusion_patterns=exclusion_patterns,
        processes=processes,
    )

    files_standard_out1, pkls_standard_out1 = get_image_and_metadata_output_paths(
        files_standard1, source_dir, target_dir, file_suffix1
    )
    files_DWI_out1, pkls_DWI_out1 = get_image_and_metadata_output_paths(files_DWI1, source_dir, target_dir, file_suffix1)
    files_PET_out1, pkls_PET_out1 = get_image_and_metadata_output_paths(files_PET1, source_dir, target_dir, file_suffix1)
    bvals_DWI1, bvecs_DWI1 = get_bvals_and_bvecs_v1(files_DWI1, file_suffix1)

    files_standard2, files_DWI2, files_PET2, files_excluded2 = detect_cases(
        source_dir,
        extension=file_suffix2,
        DWI_patterns=DWI_patterns,
        PET_patterns=PET_patterns,
        exclusion_patterns=exclusion_patterns,
        processes=processes,
    )

    files_standard_out2, pkls_standard_out2 = get_image_and_metadata_output_paths(
        files_standard2, source_dir, target_dir, file_suffix2
    )
    files_DWI_out2, pkls_DWI_out2 = get_image_and_metadata_output_paths(files_DWI2, source_dir, target_dir, file_suffix2)
    files_PET_out2, pkls_PET_out2 = get_image_and_metadata_output_paths(files_PET2, source_dir, target_dir, file_suffix2)
    bvals_DWI2, bvecs_DWI2 = get_bvals_and_bvecs_v1(files_DWI2, file_suffix2)

    files_standard = list(files_standard1) + list(files_standard2)
    files_DWI = list(files_DWI1) + list(files_DWI2)
    files_PET = list(files_PET1) + list(files_PET2)
    files_excluded = list(files_excluded1) + list(files_excluded2)
    files_standard_out = list(files_standard_out1) + list(files_standard_out2)
    files_DWI_out = list(files_DWI_out1) + list(files_DWI_out2)
    files_PET_out = list(files_PET_out1) + list(files_PET_out2)
    pkls_standard_out = list(pkls_standard_out1) + list(pkls_standard_out2)
    pkls_DWI_out = list(pkls_DWI_out1) + list(pkls_DWI_out2)
    pkls_PET_out = list(pkls_PET_out1) + list(pkls_PET_out2)
    bvals_DWI = list(bvals_DWI1) + list(bvals_DWI2)
    bvecs_DWI = list(bvecs_DWI1) + list(bvecs_DWI2)

    multiprocess_mri_dwi_pet_cases(
        files_standard=files_standard,
        files_standard_out=files_standard_out,
        pkls_standard_out=pkls_standard_out,
        files_DWI=files_DWI,
        bvals_DWI=bvals_DWI,
        bvecs_DWI=bvecs_DWI,
        files_DWI_out=files_DWI_out,
        pkls_DWI_out=pkls_DWI_out,
        files_PET=files_PET,
        files_PET_out=files_PET_out,
        pkls_PET_out=pkls_PET_out,
        preprocessing_config=GBrainPreprocessingConfig,
        strict=False,
        processes=processes,
        chunksize=10,
    )

    postprocess_standard_dataset(
        target_dir=target_dir,
        file_suffix=file_suffix1,
        task_name=task_name,
        DWI_patterns=DWI_patterns,
        PET_patterns=PET_patterns,
        exclusion_patterns=exclusion_patterns,
        source_files_standard=files_standard,
        source_files_DWI=files_DWI,
        source_files_PET=files_PET,
        source_files_excluded=files_excluded,
        preprocessing_config=GBrainPreprocessingConfig,
        processes=processes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    args = parser.parse_args()
    convert(processes=args.num_workers)
