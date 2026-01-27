#!/usr/bin/env python3

import argparse
import json
from asparagus.functional.task_conversion_and_preprocessing import enhanced_save_json
from asparagus.paths import get_data_path
from pathlib import Path
from tqdm import tqdm


def get_scratch_path():
    return Path("/scratch/asparagus/data")


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", required=True)
    args = parser.parse_args()

    data_path = Path(get_data_path())

    task_dir = data_path / args.task
    split_files = [f for f in task_dir.glob("split_*.json") if not f.name.endswith("scratch.json")]
    if len(split_files) != 1:
        raise RuntimeError(f"Expected one split file, found {len(split_files)}")

    split_file = split_files[0]
    data = load_json(split_file)

    proj_root = str(data_path.resolve())
    scratch_root = str(get_scratch_path().resolve())

    scratch_splits = []

    for split in tqdm(data):
        scratch_split = {}
        for key in ("train", "val"):
            scratch_split[key] = [p.replace(proj_root, scratch_root) for p in split[key]]
        scratch_splits.append(scratch_split)

    enhanced_save_json(scratch_splits, task_dir / f"{split_file.stem}_scratch.json")


if __name__ == "__main__":
    main()
