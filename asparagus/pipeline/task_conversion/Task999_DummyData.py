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
import torch


def convert():
    task_name = "Task999_DummyData"
    file_suffix = ".pt"

    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    for i in range(500):
        data = torch.rand(1, 64, 64, 64)
        torch.save(data, os.path.join(target_dir, f"dummy_data_{i}.pt"))

    postprocess_standard_dataset(
        target_dir=target_dir,
        file_suffix=file_suffix,
        task_name=task_name,
        DWI_patterns=[],
        PET_patterns=[],
        exclusion_patterns=[],
        source_files_standard=[],
        source_files_DWI=[],
        source_files_PET=[],
        source_files_excluded=[],
        preprocessing_config=None,
        processes=1,
    )


if __name__ == "__main__":
    convert()
