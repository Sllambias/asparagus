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
)
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.presets.preprocessing_presets import GBrainPreprocessingConfig
from itertools import repeat
from multiprocessing.pool import Pool


def convert(path: str = get_source_path(), subdir: str = "OASIS3", processes=12):
    task_name = "Task006_OASIS4"
    file_suffix = ""  # e.g. ".nii.gz" or ".nii"
    exclusion_patterns = ["rest"]  # e.g. "func" or "fmri"
    DWI_patterns = ["dwi", "DWI"]  # e.g. "DWI" or "dwi"
    PET_patterns = ["PET"]  # e.g. "PET" or "pet"

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
    bvals_DWI, bvecs_DWI = (
        [],
        [],
    )  # use e.g. get_bvals_and_bvecs_v1(files_DWI, file_suffix) to get bvals and bvecs in the same directory as the images

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

    all_files_out = detect_final_cases(target_dir, extension=".pt")
    skipped_files = len(files_standard_out) + len(files_PET) + len(files_DWI) - len(all_files_out)
    generate_dataset_json(
        join(target_dir, "dataset.json"),
        dataset_name=task_name,
        metadata={
            "file_suffix": file_suffix,
            "patterns_exclusion": exclusion_patterns,
            "patterns_DWI": DWI_patterns,
            "patterns_PET": PET_patterns,
            "files_total_in_source_directory": len(files_standard) + len(files_DWI) + len(files_PET) + len(files_excluded),
            "files_standard_in_source_directory": len(files_standard),
            "files_DWI_in_source_directory": len(files_DWI),
            "files_PET_in_source_directory": len(files_PET),
            "files_excluded_in_source_directory": len(files_excluded),
            "files_skipped_during_processing": skipped_files,
            "final_files": len(all_files_out),
        },
        preprocessing_module=GBrainPreprocessingConfig,
    )
    generate_path_json(all_files_out, join(target_dir, "paths.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    args = parser.parse_args()
    convert(processes=args.num_workers)
