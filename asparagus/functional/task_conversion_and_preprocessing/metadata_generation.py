import os
from yucca.functional.utils.saving import enhanced_save_json
from yucca.pipeline.task_conversion.utils import get_identifiers_from_splitted_files, files_in_dir
from asparagus.modules.dataclasses.preprocessing import PreprocessingConfig


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: str,
    modalities: dict,
    labels: dict,
    dataset_name: str,
    preprocessing_module: PreprocessingConfig = None,
    label_hierarchy: dict = {},
    tasks: list = [],
    dataset_description: str = "",
):
    first_file = files_in_dir(imagesTr_dir)[0]
    im_ext = os.path.split(first_file)[-1].split(os.extsep, 1)[-1]
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir, im_ext, tasks)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir, im_ext, tasks)
    else:
        test_identifiers = []

    labels = {str(i): labels[i] for i in labels.keys()} if labels is not None else None

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["image_extension"] = im_ext
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict["labels"] = labels
    json_dict["label_hierarchy"] = label_hierarchy
    json_dict["tasks"] = tasks
    json_dict["numTraining"] = len(train_identifiers)
    json_dict["numTest"] = len(test_identifiers)
    json_dict["test"] = test_identifiers
    json_dict["preprocessing_module"] = preprocessing_module

    enhanced_save_json(json_dict, os.path.join(output_file))
