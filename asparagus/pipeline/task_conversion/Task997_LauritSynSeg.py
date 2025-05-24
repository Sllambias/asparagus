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
)
from asparagus.paths import get_data_path
import torch


def convert():
    task_name = "Task997_LauritSynSeg"
    file_suffix = ".pt"

    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    torch.manual_seed(421890)
    for i in range(500):
        dims = (
            torch.randint(low=20, high=768, size=(1,)),
            torch.randint(low=20, high=768, size=(1,)),
            torch.randint(low=20, high=768, size=(1,)),
        )
        data = torch.FloatTensor(
            1,
            *dims,
        ).uniform_(-1.1e37, 1.1e37)
        seg = torch.randint(
            low=0,
            high=5,
            size=(1, *dims),
        )
        torch.save(torch.cat([data, seg]), os.path.join(target_dir, f"LauritSynSeg_{i}.pt"))

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
        n_classes=5,
        n_modalities=1,
    )


if __name__ == "__main__":
    convert()
