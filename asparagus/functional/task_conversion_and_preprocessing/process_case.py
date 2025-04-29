import numpy as np
import nibabel as nib
import os
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from dataclasses import asdict
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from batchgenerators.utilities.file_and_folder_operations import save_pickle


def process_mri_case(path, image_save_path, pkl_save_path, preprocessing_config, dtype=np.float16):
    image = nib.load(path)
    case, image_props = preprocess_case_for_training_without_label(images=[image], **asdict(preprocessing_config))
    os.makedirs(os.path.split(image_save_path)[0], exist_ok=True)
    np.save(image_save_path, np.array(case, dtype=dtype))
    save_pickle(image_props, pkl_save_path)


def process_dwi_case(path, root, output_dir, dtype=np.float16):
    raise NotImplementedError("DWI case processing is not implemented yet.")


def process_pet_case(path, root, output_dir, dtype=np.float16):
    raise NotImplementedError("PET case processing is not implemented yet.")
