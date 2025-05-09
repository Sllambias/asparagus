import argparse
from asparagus.paths import get_source_path
import importlib
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="Name of the task to preprocess. " "Should be of format: TaskXXX_MYTASK", required=True
    )
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    args = parser.parse_args()
    task_converter = importlib.import_module(f"asparagus.pipeline.task_conversion.{args.task}")
    task_converter.convert(processes=args.num_workers)


if __name__ == "__main__":
    main()
