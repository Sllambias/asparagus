from asparagus.paths import (
    get_additional_evalbox_config_path,
    get_additional_finetune_config_path,
    get_additional_pretrain_config_path,
    get_additional_train_config_path,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class FinetuneSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for path in get_additional_finetune_config_path(optional=True):
            search_path.append(provider="finetune-searchpath-plugin", path="file://" + path)


class TrainSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for path in get_additional_train_config_path(optional=True):
            search_path.append(provider="train-searchpath-plugin", path="file://" + path)


class PretrainSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for path in get_additional_pretrain_config_path(optional=True):
            search_path.append(provider="pretrain-searchpath-plugin", path="file://" + path)


class EvalBoxesSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for path in get_additional_evalbox_config_path(optional=True):
            search_path.append(provider="evalbox-searchpath-plugin", path="file://" + path)
