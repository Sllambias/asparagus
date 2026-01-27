import hydra
import logging
import os
import random
import re
import subprocess
import yaml
from asparagus.functional.loading import load_json
from asparagus.functional.scheduling import get_run_cmd_for_scheduler, get_scheduler
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import (
    EvalBoxesSearchpathPlugin,
    FinetuneSearchpathPlugin,
    TrainSearchpathPlugin,
)
from asparagus.paths import get_config_path, get_models_path
from dotenv import load_dotenv
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("version", lambda: generate_unused_run_id(), use_cache=True)
Plugins.instance().register(EvalBoxesSearchpathPlugin)
Plugins.instance().register(FinetuneSearchpathPlugin)
Plugins.instance().register(TrainSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    hydra_cfg_runtime_choices = hydra_cfg.runtime.choices
    scheduler = get_scheduler(mode=cfg.scheduler)
    env_cmd = os.environ["ASPARAGUS_EVAL_BOX_ENV_CMD"]
    print(OmegaConf.to_yaml(cfg), hydra_cfg, scheduler, env_cmd, sep="\n")

    for task in cfg.segmentation_tasks:
        if task is None:
            continue

        hardware_cfg = task.get("hardware", cfg.default_hardware)

        asparagus_cmd = (
            f"asp_finetune_seg "
            f"--config-name={task.get('task')} "
            f"checkpoint_run_id={cfg.checkpoint_run_id} "
            f"load_checkpoint_name={cfg.load_checkpoint_name} "
            f"+model={hydra_cfg_runtime_choices.model} "
            f"+hardware={hardware_cfg} "
            f"root={hydra_cfg.job.config_name} "
        )

        run_cmd = get_run_cmd_for_scheduler(scheduler, resolve_hardware_config(hardware_cfg), asparagus_cmd, env_cmd)
        logging.info(f"Running eval box for segmentation task {task} with runcommand: {' '.join(run_cmd)}")
        subprocess.run(run_cmd, capture_output=True, text=True)

    for task in cfg.classification_tasks:
        if task is None:
            continue

        hardware_cfg = task.get("hardware", cfg.default_hardware)

        asparagus_cmd = (
            f"asp_finetune_cls "
            f"--config-name={task.get('task')} "
            f"checkpoint_run_id={cfg.checkpoint_run_id} "
            f"load_checkpoint_name={cfg.load_checkpoint_name} "
            f"+model={hydra_cfg_runtime_choices.model} "
            f"+hardware={hardware_cfg} "
            f"root={hydra_cfg.job.config_name} "
        )

        run_cmd = get_run_cmd_for_scheduler(scheduler, resolve_hardware_config(hardware_cfg), asparagus_cmd, env_cmd)
        logging.info(f"Running eval box for classification task {task} with runcommand: {' '.join(run_cmd)}")
        subprocess.run(run_cmd, capture_output=True, text=True)

    for task in cfg.regression_tasks:
        if task is None:
            continue

        hardware_cfg = task.get("hardware", cfg.default_hardware)

        asparagus_cmd = (
            f"asp_finetune_reg "
            f"--config-name={task.get('task')} "
            f"checkpoint_run_id={cfg.checkpoint_run_id} "
            f"load_checkpoint_name={cfg.load_checkpoint_name} "
            f"+model={hydra_cfg_runtime_choices.model} "
            f"+hardware={hardware_cfg} "
            f"root={hydra_cfg.job.config_name} "
        )

        run_cmd = get_run_cmd_for_scheduler(scheduler, resolve_hardware_config(hardware_cfg), asparagus_cmd, env_cmd)
        logging.info(f"Running eval box for regression task {task} with runcommand: {' '.join(run_cmd)}")
        subprocess.run(run_cmd, capture_output=True, text=True)


@hydra.main(
    config_path=get_config_path(),
    config_name="",
    version_base="1.2",
)
def prepare_data(cfg: DictConfig) -> None:
    scheduler = get_scheduler(mode=cfg.scheduler)
    env_cmd = os.environ["ASPARAGUS_EVAL_BOX_ENV_CMD"]
    for config_name in cfg.segmentation_tasks + cfg.classification_tasks + cfg.regression_tasks:
        if config_name is None:
            continue
        subcfg = compose(config_name, overrides=[])
        task = subcfg.task
        asparagus_cmd = f"asp_process --dataset {task} --save_as_tensor --num_workers {cfg.hardware.num_workers} "
        run_cmd = get_run_cmd_for_scheduler(scheduler, cfg, asparagus_cmd, env_cmd)
        logging.info(f"Preparing eval box data for task: {task} with num_workers: {cfg.hardware.num_workers}")
        subprocess.run(run_cmd, capture_output=True, text=True)


@hydra.main(
    config_path=get_config_path(),
    config_name="",
    version_base="1.2",
)
def collect_results(cfg: DictConfig) -> None:
    parent_run_id = cfg.checkpoint_run_id
    parent_ckpt_name = cfg.load_checkpoint_name

    seg_cfgs, cls_cfgs, regr_cfgs = resolve_subconfigs_for_config(cfg)
    results = {"segmentation_results": [], "classification_results": [], "regression_results": []}

    for root, dirs, files in os.walk(get_models_path()):
        # print(dirs)
        # if f"stem={parent_run_id}_{parent_ckpt_name}"
        # 1) find models that have run inference
        if "predictions" in dirs:
            # 2) select those that are using correkt ckpt
            if f"stem={parent_run_id}_{parent_ckpt_name}" in root:
                # 3) select those that are trained using relevant configs
                if any([config in root for config in seg_cfgs]):
                    metadata = get_model_metadata_from_path(root)
                    inference_data = get_inference_data_for_dir(root + "/predictions", task_type="segmentation")
                    for dataset in inference_data.keys():
                        inference_data[dataset].update(metadata)
                    results["segmentation_results"].append(inference_data)

                if any([config in root for config in cls_cfgs]):
                    metadata = get_model_metadata_from_path(root)
                    inference_data = get_inference_data_for_dir(root + "/predictions", task_type="classification")
                    for dataset in inference_data.keys():
                        inference_data[dataset].update(metadata)
                    results["classification_results"].append(inference_data)

                if any([config in root for config in regr_cfgs]):
                    metadata = get_model_metadata_from_path(root)
                    inference_data = get_inference_data_for_dir(root + "/predictions", task_type="regression")
                    for dataset in inference_data.keys():
                        inference_data[dataset].update(metadata)
                    results["regression_results"].append(inference_data)

    with open("/home/zcr545/TEST.yaml", "w") as outfile:
        yaml.dump(results, outfile, sort_keys=False)


def resolve_subconfigs_for_config(config):
    seg_tasks, cls_tasks, regr_tasks = [], [], []
    for task in config.segmentation_tasks:
        if task is not None:
            seg_tasks.append(task.get("task"))
    for task in config.classification_tasks:
        if task is not None:
            cls_tasks.append(task.get("task"))
    for task in config.regression_tasks:
        if task is not None:
            regr_tasks.append(task.get("task"))
    return seg_tasks, cls_tasks, regr_tasks


def get_model_metadata_from_path(path):
    run_metadata = {"config": None, "fold": None, "runID": None}
    fold_regex = r"__fold=(\d+)"
    config_regex = r"/leaf=(\S+)__clargs"
    run_id_regex = r"/run_id=(\d+)"
    if re.search(fold_regex, path) is not None:
        run_metadata["fold"] = int(re.search(fold_regex, path).group(1))
    if re.search(config_regex, path) is not None:
        run_metadata["config"] = re.search(config_regex, path).group(1)
    if re.search(run_id_regex, path) is not None:
        run_metadata["runID"] = int(re.search(run_id_regex, path).group(1))
    return run_metadata


def get_prediction_metadata_from_path(path):
    dataset, split, ckpt = path.split("__")
    ckpt = ckpt.replace(".json", "")
    return dataset, split, ckpt


def get_segmentation_prediction_metrics_from_json(path):
    metrics_of_interest = ["dice", "volume_similarity"]
    results = {metric: [] for metric in metrics_of_interest}
    predictions = load_json(path)["mean"]
    for label in predictions.keys():
        if int(label) == 0:
            continue
        for metric in metrics_of_interest:
            results[metric].append(predictions[label][metric])
    for k, v in results.items():
        results[k] = round(sum(v) / len(v), 4)
    return results


def get_classification_prediction_metrics_from_json(path):
    metrics_of_interest = ["Precision", "Recall"]
    results = {metric: [] for metric in metrics_of_interest}
    predictions = load_json(path)["metrics"]
    for metric in metrics_of_interest:
        results[metric] = round(sum(predictions[metric]) / len(predictions[metric]), 4)
    return results


def get_regression_prediction_metrics_from_json(path):
    metrics_of_interest = ["MAE", "MSE"]
    results = {metric: [] for metric in metrics_of_interest}
    predictions = load_json(path)["metrics"]
    for metric in metrics_of_interest:
        results[metric] = round(predictions[metric], 4)
    return results


def get_inference_data_for_dir(path, task_type):
    inference_data = {}
    for file in os.listdir(path):
        if not file.endswith(".json"):
            continue
        dataset, split, ckpt = get_prediction_metadata_from_path(file)
        if task_type == "segmentation":
            metrics = get_segmentation_prediction_metrics_from_json(os.path.join(path, file))
        elif task_type == "classification":
            metrics = get_classification_prediction_metrics_from_json(os.path.join(path, file))
        elif task_type == "regression":
            metrics = get_regression_prediction_metrics_from_json(os.path.join(path, file))
        else:
            print("unexpected task type", task_type)
        inference_data[dataset] = {
            "config": None,
            "checkpoint": ckpt,
            "fold": None,
            "runID": None,
            "split": split,
            "metrics": metrics,
        }
    return inference_data


def resolve_hardware_config(hardware):
    hardware_cfg_resolved = compose("hardware/" + hardware, overrides=[])["hardware"]
    return hardware_cfg_resolved


if __name__ == "__main__":
    main()


# %%


# %%
