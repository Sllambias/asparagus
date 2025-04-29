# %%
import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
    subfiles,
    save_pickle,
)
from asparagus.functional.task_conversion_and_preprocessing.metadata_generation import generate_dataset_json
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from tqdm.contrib.concurrent import process_map
from dataclasses import asdict

Task001_preprocessing = PreprocessingConfig(normalization_operation=["volume_wise_znorm"], target_spacing=[1.0, 1.0, 1.0])


def process_case(id, input_dir, output_dir):
    image = nib.load(join(input_dir, id))
    case, image_props = preprocess_case_for_training_without_label(images=[image], **asdict(Task001_preprocessing))
    save_path = join(output_dir, id)
    np.save(save_path + ".npy", np.array(case, dtype=np.float16))
    save_pickle(image_props, save_path + ".pkl")


def convert(path: str = get_source_path(), subdir: str = "OASIS"):
    file_suffix = ".nii"
    images_dir_tr = join(path, subdir, "Images", "Train")
    training_samples = subfiles(images_dir_tr, join=False, suffix=file_suffix)

    task_name = "Task001_OASIS"

    target_dir = join(get_data_path(), task_name, "imagesTr")
    ensure_dir_exists(target_dir)

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
        ("T1",),
        labels={0: "background", 1: "Left Hippocampus", 2: "Right Hippocampus"},
        dataset_name=task_name,
        preprocessing_module=Task001_preprocessing,
    )


if __name__ == "__main__":
    convert()
