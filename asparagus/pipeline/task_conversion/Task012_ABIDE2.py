import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from asparagus.functional.task_conversion_and_preprocessing import (
    generate_dataset_json,
    generate_path_json,
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


def convert(path: str = get_source_path(), subdir: str = "ABIDE_II", processes=12):
    task_name = "Task012_ABIDE2"
    file_suffix = ".nii.gz"  # e.g. ".nii.gz" or ".nii"
    exclusion_patterns = ["rest"]  # e.g. "func" or "fmri"
    DWI_patterns = ["dti"]  # e.g. "DWI" or "dwi"
    PET_patterns = []  # e.g. "PET" or "pet"

    source_dir = join(path, subdir)
    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    files_standard, files_DWI, files_PET, files_excluded = detect_cases(
        source_dir,
        extension=file_suffix,
        DWI_patterns=DWI_patterns,
        PET_patterns=PET_patterns,
        exclusion_patterns=exclusion_patterns,
        processes=processes,
    )

    files_standard_out, pkls_standard_out = get_image_and_metadata_output_paths(
        files_standard, source_dir, target_dir, file_suffix
    )
    files_DWI_out, pkls_DWI_out = get_image_and_metadata_output_paths(files_DWI, source_dir, target_dir, file_suffix)
    files_PET_out, pkls_PET_out = get_image_and_metadata_output_paths(files_PET, source_dir, target_dir, file_suffix)
    bvals_DWI, bvecs_DWI = get_bvals_and_bvecs_v1(files_DWI, file_suffix)

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
        processes=processes,
        chunksize=10,
    )

    postprocess_standard_dataset(
        target_dir=target_dir,
        file_suffix=file_suffix,
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
