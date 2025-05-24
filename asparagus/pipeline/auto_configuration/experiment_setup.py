from hydra.core.hydra_config import HydraConfig
from asparagus.pipeline.auto_configuration import versioning, pathing
from asparagus.modules.dataclasses import TrainingFiles
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists, join, load_json
import lightning as pl


def prepare_standard_experiment(cfg):
    pathingcfg = pathing(cfg)
    versioncfg = versioning(cfg)
    filecfg = TrainingFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path), splits=load_json(cfg._internal_.splits_path)[cfg.data.fold]
    )
    return filecfg, pathingcfg, versioncfg
