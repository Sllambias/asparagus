import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from asparagus.functional.task_conversion_and_preprocessing import (
    get_image_and_metadata_output_paths,
    get_bvals_and_bvecs_v1,
    multiprocess_mri_dwi_pet_cases,
    postprocess_standard_dataset,
    process_numpy_case,
    preprocess_case_for_training_without_label,
)
from asparagus.paths import get_data_path
import torch
from multiprocessing.pool import Pool
from asparagus.modules.dataclasses.presets.preprocessing_presets import ClsPreprocessingConfig
from dataclasses import asdict
from itertools import repeat
import numpy as np


def convert(processes=6):
    task_name = "Task996_LauritSynCls"
    file_suffix = ".pt"
    n_classes = 4
    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    torch.manual_seed(421890)

    p = Pool(processes)
    p.starmap_async(generate_random_clscase, zip(range(500), repeat(target_dir), repeat(n_classes)))
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
        preprocessing_config=ClsPreprocessingConfig,
        processes=1,
        n_classes=5,
        n_modalities=1,
    )


def generate_random_clscase(i, target_dir, n_classes):
    dims = (
        torch.randint(low=60, high=280, size=(1,)),
        torch.randint(low=60, high=280, size=(1,)),
        torch.randint(low=60, high=280, size=(1,)),
    )
    data = torch.FloatTensor(*dims).uniform_(-1.1e16, 1.1e16).numpy()

    label = torch.randint(
        low=0,
        high=n_classes,
        size=(1,),
    )

    images, properties = preprocess_case_for_training_without_label(
        images=[data],
        **asdict(ClsPreprocessingConfig),
        strict=False,
    )

    torch.save(
        [torch.tensor(images), label],
        os.path.join(target_dir, f"LauritSynCls_{i}.pt"),
    )


if __name__ == "__main__":
    convert()
