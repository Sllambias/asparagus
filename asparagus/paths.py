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
