import logging
import numpy as np
import os
import yaml
from asparagus.paths import get_models_path
from gardening_tools.functional.paths.scan import subfiles
from typing import Union


def generate_unused_run_id(
    resume_training: bool, run_dir: str = None, model_dir: str = get_models_path(), string_match="run_id="
) -> Union[None, int]:
    if resume_training and os.path.isdir(run_dir):
        previous_runs = detect_previous_runs_with_ckpt(run_dir=run_dir, ckpt="last")
        if len(previous_runs) > 0:
            return get_id_for_most_recent_ckpt(run_dir, previous_runs, string_match=string_match, ckpt="last")
    # get list of all run ids used by asparagus
    global_used_ids = [0]
    for dirpath, _, _ in os.walk(model_dir):
        d = os.path.split(dirpath)[1]
        if d.startswith(string_match):
            global_used_ids.append(int(d.replace(string_match, "")))

    # find a globally unique one
    run_id = 0
    while run_id in global_used_ids:
        run_id = int(np.random.randint(1000, 999999))

    return run_id


def get_id_for_most_recent_ckpt(run_dir, previous_runs, string_match="run_id=", ckpt: str = "last"):
    ctime = 0
    id = ""
    for run in previous_runs:
        run_ctime = os.path.getctime(os.path.join(run_dir, run, f"checkpoints/{ckpt}.ckpt"))
        if ctime < run_ctime:
            id = run
            ctime = run_ctime

    return id.replace(string_match, "")


def detect_previous_runs_with_ckpt(run_dir, ckpt: str = "last"):
    previous_runs = os.listdir(run_dir)
    valid_runs = []
    for run in previous_runs:
        if os.path.isfile(os.path.join(run_dir, run, f"checkpoints/{ckpt}.ckpt")):
            valid_runs.append(run)
    return valid_runs


def detect_wandb_id(run_dir) -> Union[None, str]:
    wandb_log_dir = os.path.join(run_dir, "wandb", "latest-run")
    if not os.path.isdir(wandb_log_dir):
        return None
    files = subfiles(wandb_log_dir, suffix=".wandb", join=False)
    if not len(files) > 0:
        return None
    id = files[0].replace("run-", "").replace(".wandb", "")
    return id


def detect_mlflow_id(run_dir: str) -> Union[None, int]:
    mlruns_dir = os.path.join(run_dir, "mlruns")

    run_id = None
    max_start_time = 0

    for root, _, files in os.walk(mlruns_dir):
        if "meta.yaml" in files:
            meta_yaml_path = os.path.join(root, "meta.yaml")

            with open(meta_yaml_path, "r") as f:
                meta_data = yaml.safe_load(f)

            id = meta_data.get("run_id")
            start_time = meta_data.get("start_time", 0)

            # take the run which was started last if there are multiple...
            if start_time > max_start_time:
                run_id = id

    return run_id


def detect_id(id: str, model_dir: str = get_models_path()):
    id = str(id)
    # CASE SENSITIVE
    all_cases = []
    # 1. Build list of all files in the directory recursively.
    for dirpath, _, _ in os.walk(model_dir):
        if dirpath.endswith(f"run_id={id}"):
            all_cases.append(dirpath)
    # 2. Check if the ID is in the list of files.
    if len(all_cases) == 1:
        logging.info(f"Found exactly 1 match for ID: {id} in {model_dir}")
    else:
        raise ValueError(f"Found {len(all_cases)} matches for ID: {id} in {model_dir}")

    return all_cases[0]


if __name__ == "__main__":
    print(detect_id("582109"))
