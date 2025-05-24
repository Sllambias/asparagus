from typing import Union
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
    join,
    subdirs,
    isdir,
    subfiles,
    isfile,
)
from asparagus.paths import get_models_path
import os
import logging


def detect_version(save_dir, continue_from_most_recent) -> Union[None, int]:
    # If the dir doesn't exist we return version 0
    if not isdir(save_dir):
        return 0

    # The dir exists. Check if any previous version exists in dir.
    previous_versions = subdirs(save_dir, prefix="version", join=False)
    # If no previous version exists we return version 0
    if not previous_versions:
        return 0

    # If previous version(s) exists we can either (1) continue from the newest or
    # (2) create the next version
    if previous_versions:
        newest_version = max([int(i.split("_")[-1]) for i in previous_versions])
        if continue_from_most_recent:
            return newest_version
        else:
            return newest_version + 1


def detect_wandb_id(version_dir, continue_from_most_recent) -> Union[None, str]:
    if not continue_from_most_recent:
        return None
    wandb_log_dir = join(version_dir, "wandb", "latest-run")
    if not isdir(wandb_log_dir):
        return None
    files = subfiles(wandb_log_dir, suffix=".wandb", join=False)
    if not len(files) > 0:
        return None
    id = files[0].replace("run-", "").replace(".wandb", "")
    return id


def detect_id(id: str, model_dir: str = get_models_path()):
    id = str(id)
    # CASE SENSITIVE
    all_cases = []
    # 1. Build list of all files in the directory recursively.
    for dirpath, _, _ in os.walk(model_dir):
        if dirpath.endswith(f"id={id}"):
            all_cases.append(dirpath)
    if len(all_cases) < 1:
        logging.warning(f"found 0 matches for ID: {id} in {model_dir}")
    if len(all_cases) > 1:
        logging.warning(f"found multiple matches for ID: {id} in {model_dir}")
    if len(all_cases) == 1:
        logging.info(f"found exactly 1 match for ID: {id} in {model_dir}")
    return all_cases


if __name__ == "__main__":
    print(detect_id("582109"))
