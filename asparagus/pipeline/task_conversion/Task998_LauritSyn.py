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
from asparagus.paths import get_data_path
from asparagus.modules.dataclasses.presets.preprocessing_presets import GBrainPreprocessingConfig
import torch
from multiprocessing.pool import Pool
from itertools import repeat


def convert(processes=6):
    task_name = "Task998_LauritSyn"
    file_suffix = ".pt"

    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    torch.manual_seed(421890)

    p = Pool(processes)
    p.starmap_async(generate_random_case, zip(range(500), repeat(target_dir)))
    p.close()
    p.join()

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


def generate_random_case(i, target_dir):
    dims = (
        1,
        torch.randint(low=20, high=768, size=(1,)),
        torch.randint(low=20, high=768, size=(1,)),
        torch.randint(low=20, high=768, size=(1,)),
    )
    data = torch.FloatTensor(*dims).uniform_(-1.1e37, 1.1e37)
    torch.save(data, os.path.join(target_dir, f"LauritSyn_{i}.pt"))


if __name__ == "__main__":
    convert()
