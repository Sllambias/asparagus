import glob
import numpy as np
import os
import pathlib
import shutil
from asparagus.paths import get_models_path


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


def prune_model_dirs(root_model_dir, dry_run=True):
    print(f"Dry run: {dry_run}")
    for path in glob.glob(root_model_dir + "/**/*", recursive=True):
        if not os.path.isdir(path):
            continue
        if "run_id=" not in os.path.split(path)[-1]:
            continue
        if "checkpoints" in os.listdir(path):
            print(f"Found model in: {path}")
            continue
        if "predictions" in os.listdir(path):
            print(f"Found predictions in: {path}")
            continue
        if not all(subpath.endswith((".log", ".txt", ".yaml", "hydra", "mlruns", "wandb")) for subpath in os.listdir(path)):
            print("skipping", path)

        print(f"Removing: {path}")
        if not dry_run:
            shutil.rmtree(path)

    if not dry_run:
        for _ in range(10):
            for path in glob.glob(root_model_dir + "/**/*", recursive=True):
                if os.path.isdir(path):
                    if len(os.listdir(path)) == 0:
                        print("Removing empty dir: ", path)
                        pathlib.Path.rmdir(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prune-model-dir", action="store_true")
    parser.add_argument("--model-dir", default=get_models_path())
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    if args.prune_model_dir:
        prune_model_dirs(root_model_dir=args.model_dir, dry_run=not args.fix)
