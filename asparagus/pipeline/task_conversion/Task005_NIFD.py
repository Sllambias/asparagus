import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
    subfiles,
    save_pickle,
)
from asparagus.functional.task_conversion_and_preprocessing.metadata_generation import (
    generate_dataset_json,
    generate_path_json,
)
from asparagus.functional.task_conversion_and_preprocessing.detect import detect_cases
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from tqdm.contrib.concurrent import process_map
from dataclasses import asdict
import pandas as pd
import tqdm

Task001_preprocessing = PreprocessingConfig(
    normalization_operation=["volume_wise_znorm"], target_spacing=None, target_orientation="RAS"
)


def convert(path: str = get_source_path(), subdir: str = "NIFD", dry_run=True):
    task_name = "Task005_NIFD"
    file_suffix = ".dcm"
    exclusion_patterns = ["fmri"]

    target_dir = join(get_data_path(), task_name, "imagesTr")
    ensure_dir_exists(target_dir)

    metadata = pd.read_csv("/home/zcr545/data/data/public_datasets/NIFD/NIFD_4_25_2025.csv")
    PET_cases = metadata[metadata["Modality"] == "PET"]["Description"].tolist()
    DWI_cases = metadata[metadata["Modality"] == "DTI"]["Description"].tolist()

    print(f"Metadata indicates PET cases: {len(PET_cases)} and DWI cases: {len(DWI_cases)} in {len(metadata)} total cases.")
    print(f"Starting filtering process now. Excluding the following patterns entirely: {exclusion_patterns}")

    regular_files, DWI_files, PET_files, excluded_files = detect_cases(
        "/home/zcr545/data/data/projects/GBrains/NIFD",
        extension=file_suffix,
        DWI_patterns=DWI_cases,
        PET_patterns=PET_cases,
        exclusion_patterns=exclusion_patterns,
        processes=12,
    )
    print(
        f"Found {len(regular_files)} regular files, {len(DWI_files)} DWI files, {len(PET_files)} PET files, and {len(excluded_files)} excluded files."
    )
    if dry_run:
        print("DRY RUN: No files will be processed.")
        return

    process_map(
        process_case,
        training_samples,
        [images_dir_tr] * len(training_samples),
        [target_dir] * len(training_samples),
        max_workers=2,
    )

    generate_dataset_json(
        join(target_dir, "dataset.json"),
        target_dir,
        None,
        ("MRI",),
        labels=None,
        dataset_name=task_name,
        preprocessing_module=Task001_preprocessing,
    )
    generate_path_json(included_files, join(target_dir, "paths.json"))


if __name__ == "__main__":
    convert()
