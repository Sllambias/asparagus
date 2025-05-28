import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from asparagus.functional.task_conversion_and_preprocessing import (
    detect_cases,
    get_image_and_metadata_output_paths,
    multiprocess_image_label_cases,
    postprocess_standard_dataset,
)
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.presets.preprocessing_presets import GBrainPreprocessingConfig
from itertools import repeat
from multiprocessing.pool import Pool


def convert(path: str = get_source_path(), subdir: str = "OASIS", processes=12):
    task_name = "Task600_OASISMini"
    file_suffix = ".nii"
    exclusion_patterns = ["Labels"]
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
    files_label = [path.replace("Images", "Labels") for path in files_standard]
    print(files_standard_out)
    print(files_label)
    multiprocess_image_label_cases(
        files_standard=files_standard,
        files_label=files_label,
        files_standard_out=files_standard_out,
        pkls_standard_out=pkls_standard_out,
        preprocessing_config=GBrainPreprocessingConfig,
        strict=False,
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
        n_classes=3,
        n_modalities=1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    args = parser.parse_args()
    convert(processes=args.num_workers)
