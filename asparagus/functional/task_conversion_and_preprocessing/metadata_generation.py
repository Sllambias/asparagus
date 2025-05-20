import os
import json
import dataclasses
from yucca.pipeline.task_conversion.utils import get_identifiers_from_splitted_files, files_in_dir
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig
from asparagus.functional.task_conversion_and_preprocessing.detect import detect_cases, detect_final_cases
from batchgenerators.utilities.file_and_folder_operations import join


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


def postprocess_standard_dataset(
    target_dir,
    task_name,
    file_suffix,
    DWI_patterns,
    PET_patterns,
    exclusion_patterns,
    source_files_standard,
    source_files_DWI,
    source_files_PET,
    source_files_excluded,
    preprocessing_config,
    processes=12,
):

    source_all_files = len(source_files_standard) + len(source_files_DWI) + len(source_files_PET) + len(source_files_excluded)
    target_all_files = detect_final_cases(target_dir, extension=".pt")

    files_delta = len(target_all_files) - source_all_files

    target_files_standard, target_files_DWI, target_files_PET, _ = detect_cases(
        target_dir,
        extension=".pt",
        DWI_patterns=DWI_patterns,
        PET_patterns=PET_patterns,
        exclusion_patterns=exclusion_patterns,
        processes=processes,
    )
    generate_dataset_json(
        join(target_dir, "dataset.json"),
        dataset_name=task_name,
        metadata={
            "file_suffix": file_suffix,
            "patterns_exclusion": exclusion_patterns,
            "patterns_DWI": DWI_patterns,
            "patterns_PET": PET_patterns,
            "files_source_directory_total": source_all_files,
            "files_source_directory_standard": len(source_files_standard),
            "files_source_directory_DWI": len(source_files_DWI),
            "files_source_directory_PET": len(source_files_PET),
            "files_source_directory_excluded": len(source_files_excluded),
            "files_target_directory_total": len(target_all_files),
            "files_target_directory_standard": len(target_files_standard),
            "files_target_directory_DWI": len(target_files_DWI),
            "files_target_directory_PET": len(target_files_PET),
            "files_delta_after_processing": files_delta,
        },
        preprocessing_module=preprocessing_config,
    )
    enhanced_save_json(target_all_files, join(target_dir, "paths.json"))
