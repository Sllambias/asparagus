import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from asparagus.functional.task_conversion_and_preprocessing import (
    generate_dataset_json,
    process_mri_case,
    detect_cases,
    detect_final_cases,
    get_image_and_metadata_output_paths,
    postprocess_standard_dataset,
)
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.presets.preprocessing_presets import GBrainPreprocessingConfig
from itertools import repeat
from multiprocessing.pool import Pool


def convert(path: str = get_source_path(), subdir: str = "OpenMind", processes=12):
    task_name = "Task002_OpenMind"
    file_suffix = ".nii.gz"
    exclusion_patterns = ["fmri", "mask"]
    DWI_patterns = []
    PET_patterns = []

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

    p = Pool(processes)
    p.starmap_async(
        process_mri_case,
        zip(
            files_standard,
            files_standard_out,
            pkls_standard_out,
            repeat(GBrainPreprocessingConfig),
        ),
        chunksize=25,
    )
    p.close()
    p.join()

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
