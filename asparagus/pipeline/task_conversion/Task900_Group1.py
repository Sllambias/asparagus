import os
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, load_json
from asparagus.functional.task_conversion_and_preprocessing import (
    generate_dataset_json,
    generate_path_json,
    process_mri_case,
    detect_cases,
    detect_final_cases,
    detect_task_name_from_task_id,
)
from asparagus.paths import get_data_path, get_source_path
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from dataclasses import asdict
from itertools import repeat
from multiprocessing.pool import Pool
import pandas as pd


def convert(path: str = get_source_path(), subdir: str = ""):
    task_name = "Task900_Group1"
    sub_tasks = [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
        "010",
        "011",
        "012",
        "013",
        "014",
        "015",
        "016",
        "017",
        "018",
        "019",
        "020",
        "021",
        "022",
        "023",
        "024",
        "025",
        "026",
        "027",
        "028",
        "029",
        "030",
        "031",
        "032",
        "033",
    ]

    target_dir = join(get_data_path(), task_name)
    ensure_dir_exists(target_dir)

    all_dataset_json = {}
    all_files_out = []

    for task in sub_tasks:
        task = detect_task_name_from_task_id(task)
        task_dir = join(get_data_path(), task)
        dataset_json = load_json(join(task_dir, "dataset.json"))
        paths_json = load_json(join(task_dir, "paths.json"))
        all_dataset_json[task] = dataset_json
        all_files_out += paths_json

    generate_dataset_json(
        join(target_dir, "dataset.json"),
        dataset_name=task_name,
        metadata={
            "sub dataset_jsons": all_dataset_json,
            "sub tasks": sub_tasks,
            "final_files": len(all_files_out),
        },
    )
    generate_path_json(all_files_out, join(target_dir, "paths.json"))


if __name__ == "__main__":
    convert()
