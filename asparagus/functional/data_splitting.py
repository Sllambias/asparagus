from asparagus.functional.task_conversion_and_preprocessing import enhanced_save_json
from sklearn.model_selection import train_test_split


def split_80_20(files: list):
    train, val = train_test_split(files, test_size=0.2, random_state=963421)
    return train, val


def split(files: list, fn: split_80_20, save_path: str = None):
    train, val = fn(files)
    if save_path is not None:
        enhanced_save_json(obj={"train": train, "val": val}, file=save_path)
    return train, val
