import logging
import os
from asparagus.modules.dataclasses import DataFiles
from asparagus.pipeline.auto_configuration.versioning import pathing, versioning
from gardening_tools.functional.paths.read import load_json
from hydra.utils import instantiate


def prepare_standard_experiment(cfg):
    pathingcfg = pathing(cfg, train=True)
    versioncfg = versioning(cfg)
    if not os.path.isfile(cfg.test_split_path):
        logging.warn("No test split found")
        test = None
    else:
        test = load_json(cfg.test_split_path)

    filecfg = DataFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path),
        splits=load_json(cfg.train_split_path)[cfg.data.fold],
        test=test,
    )
    logging.warning(f"###RUN-ID={versioncfg.version}###")
    return filecfg, pathingcfg, versioncfg


def prepare_inference(cfg):
    pathingcfg = pathing(cfg, train=False)
    filecfg = DataFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path),
        splits=None,
        test=load_json(cfg.data.test_split_path),
    )
    return filecfg, pathingcfg


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
    cpu_transforms = instantiate(cfg.plugins.seg._cpu_transforms)
    seg_data_module = instantiate(
        cfg.plugins.seg._data_module,
        train_split=splits["train"],
        val_split=splits["val"],
        train_transforms=cpu_transforms,
        val_transforms=cpu_transforms,
    )

    model = instantiate(
        cfg.model._plugin_seg_net,
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
