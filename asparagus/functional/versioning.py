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
import numpy as np


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


def detect_used_ids(
    save_dir, model_dir: str = get_models_path(), continue_from_most_recent: bool = True, string_match="run_id="
) -> Union[None, int]:
    local_used_ids = subdirs(save_dir, prefix=string_match, join=False)
    if len(local_used_ids) > 0:
        local_used_ids = [int(id.replace(string_match, "")) for id in local_used_ids]
    if continue_from_most_recent and len(local_used_ids) > 0:
        local_used_ids.sort(reverse=True)
        return local_used_ids[0]

    local_used_ids.append(0)
    global_used_ids = [0]
    # 1. Build list of all files in the directory recursively.
    for dirpath, _, _ in os.walk(model_dir):
        d = os.path.split(dirpath)[1]
        if d.startswith(string_match):
            global_used_ids.append(int(d.replace(string_match, "")))

    run_id = 0
    while run_id in global_used_ids or run_id <= local_used_ids[0]:
        run_id = np.random.randint(low=local_used_ids[0], high=max(100000, local_used_ids[0] * 2))

    return run_id


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
    return all_cases[0]


if __name__ == "__main__":
    print(detect_id("582109"))
