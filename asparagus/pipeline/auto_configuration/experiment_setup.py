from hydra.core.hydra_config import HydraConfig
from asparagus.pipeline.auto_configuration import versioning, pathing
from asparagus.modules.dataclasses import TrainingFiles, SegmentationPlugin
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists, join, load_json


def prepare_standard_experiment(cfg):
    pathingcfg = pathing(cfg)
    versioncfg = versioning(cfg)
    filecfg = TrainingFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path), splits=load_json(cfg._internal_.splits_path)[cfg.data.fold]
    )
    return filecfg, pathingcfg, versioncfg


def prepare_online_segmentation(cfg):
    dataset_json_path = cfg.plugins.seg.data.data_path + "/dataset.json"

    dataset_json = load_json(dataset_json_path)
    splits = load_json(cfg.plugins.seg._internal_.splits_path)[cfg.plugins.seg.data.fold]

    num_classes = dataset_json["metadata"]["n_classes"]
    num_modalities = dataset_json["metadata"]["n_modalities"]

    plugin = SegmentationPlugin(
        dataset_json=dataset_json, splits=splits, num_classes=num_classes, num_modalities=num_modalities
    )

    return plugin
