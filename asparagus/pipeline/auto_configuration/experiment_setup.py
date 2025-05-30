from asparagus.pipeline.auto_configuration import versioning, pathing
from asparagus.modules.dataclasses import TrainingFiles
from batchgenerators.utilities.file_and_folder_operations import load_json
from hydra.utils import instantiate
import logging


def prepare_standard_experiment(cfg):
    pathingcfg = pathing(cfg)
    versioncfg = versioning(cfg)
    filecfg = TrainingFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path), splits=load_json(cfg.splits_path)[cfg.data.fold]
    )
    logging.warn(f"###RUN-ID={versioncfg.version}###")
    return filecfg, pathingcfg, versioncfg


def prepare_ssl_plugins(cfg):
    plugins = []
    if cfg.plugins is None:
        return plugins
    if cfg.plugins.seg is not None:
        plugins.append(prepare_online_segmentation(cfg))
    return plugins


def prepare_online_segmentation(cfg):
    dataset_json = load_json(cfg.plugins.seg.dataset_json_path)
    splits = load_json(cfg.plugins.seg.splits_path)[cfg.plugins.seg.data.fold]

    num_classes = dataset_json["metadata"]["n_classes"]
    num_modalities = dataset_json["metadata"]["n_modalities"]

    seg_data_module = instantiate(
        cfg.plugins.seg._data_module,
        train_split=splits["train"],
        val_split=splits["val"],
        composed_train_transforms=None,
        composed_val_transforms=None,
    )

    model = instantiate(
        cfg.plugins.seg._model,
        input_channels=num_modalities,
        output_channels=num_classes,
    )

    plugin = instantiate(
        cfg.plugins.seg._plugin,
        model=model,
        data_module=seg_data_module,
        output_channels=num_classes,
    )

    return plugin
