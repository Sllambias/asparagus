import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from asparagus.functional.task_conversion_and_preprocessing import (
    generate_dataset_json,
    generate_path_json,
    process_mri_case,
    detect_cases,
)
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from tqdm.contrib.concurrent import process_map
from dataclasses import asdict
from itertools import repeat


def convert(path: str = get_source_path(), subdir: str = "ABIDE", processes=12):
    task_name = "Task001_ABIDE1"
    file_suffix = ".nii"
    exclusion_patterns = ["fmri"]

    source_dir = join(path, subdir)
    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    regular_files, DWI_files, PET_files, excluded_files = detect_cases(
        source_dir,
        extension=file_suffix,
        DWI_patterns=[],
        PET_patterns=[],
        exclusion_patterns=exclusion_patterns,
        processes=processes,
    )

    preprocessing_config = PreprocessingConfig(
        normalization_operation=["volume_wise_znorm"], target_spacing=None, target_orientation="RAS"
    )

    regular_files_out = [f.replace(source_dir, target_dir).replace(file_suffix, ".npy") for f in regular_files]
    regular_pkls_out = [f.replace(".npy", ".pkl") for f in regular_files_out]

    process_map(
        process_mri_case,
        regular_files,
        regular_files_out,
        regular_pkls_out,
        repeat(preprocessing_config),
        max_workers=processes,
        chunksize=1,
    )

    generate_dataset_json(
        join(target_dir, "dataset.json"),
        dataset_name=task_name,
        metadata={
            "file_suffix": file_suffix,
            "exclusion_patterns": exclusion_patterns,
            "number_of_regular_files": len(regular_files),
            "number_of_DWI_files": len(DWI_files),
            "number_of_PET_files": len(PET_files),
            "number_of_excluded_files": len(excluded_files),
        },
        preprocessing_module=preprocessing_config,
    )
    all_files_out = regular_files_out
    generate_path_json(all_files_out, join(target_dir, "paths.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    args = parser.parse_args()
    convert(processes=args.num_workers)
