import argparse
from asparagus.paths import get_data_path
from batchgenerators.utilities.file_and_folder_operations import load_json, join
import re
from asparagus.functional import split, all_split_fn
import asparagus.functional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="Name of the task to preprocess. Should be of format: TaskXXX_MYTASK", required=True
    )
    parser.add_argument("--fn", help=f"Splitting method to use. Choose from functions: {all_split_fn}")
    args = parser.parse_args()

    files = load_json(join(get_data_path(), args.task, "paths.json"))
    save_path = join(get_data_path(), args.task, args.fn + ".json")
    split(files=files, fn=getattr(asparagus.functional, args.fn), save_path=save_path)


if __name__ == "__main__":
    main()
