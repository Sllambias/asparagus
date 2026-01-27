import numpy as np
import os


def add_run_to_pretrained_derivative_list(version, ckpt_path, finetune_path):
    if "/version_" in ckpt_path:
        ckpt_path = ckpt_path.split("/version_")[0]
        ckpt_path = os.path.join(ckpt_path, "derived_models.log")
        with open(ckpt_path, "a+") as f:
            f.write(f"{version}, {finetune_path} \n")


def fit_patch_size_to_image_size(patch_size, image_size):
    if len(patch_size) == 2 and len(image_size) == 3:
        patch_size = np.min([patch_size, image_size[1:]], axis=0)
    else:
        patch_size = np.min([patch_size, image_size], axis=0)

    return [int(32 * (i // 32)) for i in patch_size]


def fit_image_to_patch_size(patch_size, image_size):
    if len(patch_size) == 2 and len(image_size) == 3:
        return np.max([patch_size, image_size[1:]], axis=0)
    else:
        return np.max([patch_size, image_size], axis=0)
