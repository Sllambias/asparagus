from asparagus.functional.task_conversion_and_preprocessing import enhanced_save_json
from sklearn.model_selection import train_test_split


def split_80_20(files: list):
    train, val = train_test_split(files, test_size=0.2, random_state=963421)
    return train, val


def split(files: list, fn: split_80_20, folds=5, save_path: str = None):
    splits = []
    for i in folds:
        train, val = fn(files)
        splits.append({"train": train, "val": val})
    if save_path is not None:
        enhanced_save_json(obj=splits, file=save_path)
    return train, val
