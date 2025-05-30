# %%
import os
from dotenv import load_dotenv

from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)


def var_is_set(var):
    return var in os.environ.keys()


def get_environment_variable(var):
    load_dotenv()
    if not var_is_set(var):
        raise ValueError(f"Missing required environment variable {var}.")

    path = os.environ[var]
    ensure_dir_exists(path)
    return path


def get_environment_variables(var):
    load_dotenv()
    if not var_is_set(var):
        raise ValueError(f"Missing required environment variable {var}.")

    paths = os.environ[var].split(":")
    for path in paths:
        ensure_dir_exists(path)
    return paths


def get_source_path():
    return get_environment_variable("ASPARAGUS_SOURCE")


def get_data_path():
    return get_environment_variable("ASPARAGUS_DATA")


def get_models_path():
    return get_environment_variable("ASPARAGUS_MODELS")


def get_results_path():
    return get_environment_variable("ASPARAGUS_RESULTS")


def get_config_path():
    return get_environment_variable("ASPARAGUS_CONFIGS")


def get_additional_pretrain_config_path():
    return get_environment_variables("ASPARAGUS_PRETRAIN_CONFIGS")


def get_additional_train_config_path():
    return get_environment_variables("ASPARAGUS_TRAIN_CONFIGS")


def get_additional_finetune_config_path():
    return get_environment_variables("ASPARAGUS_FINETUNE_CONFIGS")
