import os
import json
import dataclasses
from yucca.pipeline.task_conversion.utils import get_identifiers_from_splitted_files, files_in_dir
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def enhanced_save_json(
    obj, file: str, indent: int = 4, sort_keys: bool = True, cls: json.JSONEncoder = EnhancedJSONEncoder
) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, cls=cls)


def generate_dataset_json(
    output_file: str,
    dataset_name: str,
    metadata: dict = {},
    preprocessing_module: PreprocessingConfig = None,
):
    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["metadata"] = metadata
    json_dict["preprocessing_module"] = preprocessing_module

    enhanced_save_json(json_dict, os.path.join(output_file))


def generate_path_json(paths, outpath):
    enhanced_save_json(paths, outpath)
