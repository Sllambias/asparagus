import hydra
import logging
import numpy as np
import os
import random
import re
import subprocess
import yaml
from asparagus.functional.scheduling import get_run_cmd_for_scheduler, get_scheduler
from asparagus.functional.utils import find_run_dirs
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import (
    EvalBoxesSearchpathPlugin,
    FinetuneSearchpathPlugin,
    TrainSearchpathPlugin,
)
from asparagus.paths import get_config_path, get_models_path
from dotenv import load_dotenv
from gardening_tools.functional.paths.read import load_json
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver(
    "version",
    lambda resume_training, run_dir: generate_unused_run_id(resume_training=resume_training, run_dir=run_dir),
    use_cache=True,
)
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

    for task in cfg.get("segmentation_tasks") or []:
        if task is None:
            continue

        hardware_cfg = task.get("hardware", cfg.default_hardware)
        test_checkpoint = task.get("test_checkpoint", cfg.get("test_checkpoint", "best"))

        asparagus_cmd = (
            f"asp_finetune_seg "
            f"--config-name={task.get('task')} "
            f"checkpoint_run_id={cfg.checkpoint_run_id} "
            f"load_checkpoint_name={cfg.load_checkpoint_name} "
            f"test_checkpoint={test_checkpoint} "
            f"+model={hydra_cfg_runtime_choices.model} "
            f"+hardware={hardware_cfg} "
            f"root={hydra_cfg.job.config_name} "
        )

        run_cmd = get_run_cmd_for_scheduler(scheduler, resolve_hardware_config(hardware_cfg), asparagus_cmd, env_cmd)
        logging.info(f"Running eval box for segmentation task {task} with runcommand: {' '.join(run_cmd)}")
        subprocess.run(run_cmd, capture_output=True, text=True)

    for task in cfg.get("classification_tasks") or []:
        if task is None:
            continue

        hardware_cfg = task.get("hardware", cfg.default_hardware)
        test_checkpoint = task.get("test_checkpoint", cfg.get("test_checkpoint", "best"))

        asparagus_cmd = (
            f"asp_finetune_cls "
            f"--config-name={task.get('task')} "
            f"checkpoint_run_id={cfg.checkpoint_run_id} "
            f"load_checkpoint_name={cfg.load_checkpoint_name} "
            f"test_checkpoint={test_checkpoint} "
            f"+model={hydra_cfg_runtime_choices.model} "
            f"+hardware={hardware_cfg} "
            f"root={hydra_cfg.job.config_name} "
        )

        run_cmd = get_run_cmd_for_scheduler(scheduler, resolve_hardware_config(hardware_cfg), asparagus_cmd, env_cmd)
        logging.info(f"Running eval box for classification task {task} with runcommand: {' '.join(run_cmd)}")
        subprocess.run(run_cmd, capture_output=True, text=True)

    for task in cfg.get("regression_tasks") or []:
        if task is None:
            continue

        hardware_cfg = task.get("hardware", cfg.default_hardware)
        test_checkpoint = task.get("test_checkpoint", cfg.get("test_checkpoint", "best"))

        asparagus_cmd = (
            f"asp_finetune_reg "
            f"--config-name={task.get('task')} "
            f"checkpoint_run_id={cfg.checkpoint_run_id} "
            f"load_checkpoint_name={cfg.load_checkpoint_name} "
            f"test_checkpoint={test_checkpoint} "
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
    for config_name in (
        (cfg.get("segmentation_tasks") or []) + (cfg.get("classification_tasks") or []) + (cfg.get("regression_tasks") or [])
    ):
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
    parent_run_id = cfg.get("checkpoint_run_id")
    parent_ckpt_name = cfg.load_checkpoint_name
    box_config_name = HydraConfig.get().job.config_name
    # No checkpoint_run_id given -> "all models tuned with this box": report every source
    # checkpoint, keyed by checkpoint_run_id. With one given, behaviour is unchanged.
    multi_checkpoint = parent_run_id is None

    seg_cfgs, cls_cfgs, regr_cfgs = resolve_subconfigs_for_config(cfg)
    config_to_type = {}
    for config in seg_cfgs:
        config_to_type[config] = "segmentation"
    for config in cls_cfgs:
        config_to_type[config] = "classification"
    for config in regr_cfgs:
        config_to_type[config] = "regression"

    results_key = {
        "segmentation": "segmentation_results",
        "classification": "classification_results",
        "regression": "regression_results",
    }
    results = {key: [] for key in results_key.values()}

    for run_dir in find_run_dirs(get_models_path()):
        predictions_dir = os.path.join(run_dir, "predictions")

        metadata = get_run_metadata_from_hydra(run_dir)
        if metadata is None:
            continue
        # Keep only runs finetuned from the checkpoint this box is evaluating. When no
        # checkpoint_run_id was given, keep them all (grouped per checkpoint below).
        if parent_run_id is not None and str(metadata.get("checkpoint_run_id")) != str(parent_run_id):
            continue
        if metadata.get("load_checkpoint_name") != parent_ckpt_name:
            continue
        # Keep only runs whose config belongs to this box; that also gives the task type.
        config = metadata.get("config")
        if config not in config_to_type:
            continue

        run_root = metadata.get("root")
        if box_config_name and run_root is not None and run_root != box_config_name:
            continue
        task_type = config_to_type[config]

        run_metadata = {"config": config, "fold": metadata.get("fold"), "runID": metadata.get("runID")}
        if multi_checkpoint:
            run_metadata["checkpoint_run_id"] = metadata.get("checkpoint_run_id")
        inference_data = get_inference_data_for_dir(predictions_dir, task_type=task_type)
        for dataset in inference_data.keys():
            inference_data[dataset].update(run_metadata)
        results[results_key[task_type]].append(inference_data)

    # mean over folds -> mean over models -> L3 mean over datasets (+ weighted final per type).
    # With no checkpoint_run_id, do all of this PER source checkpoint and key the results by
    # checkpoint_run_id; with one given, keep the original flat single-checkpoint shape.
    weight_for_config = build_weight_map(cfg)
    aggregated, per_dataset, summary, final_scores = {}, {}, {}, {}
    for task_type, key in results_key.items():
        if multi_checkpoint:
            aggregated[task_type], per_dataset[task_type] = [], {}
            summary[task_type], final_scores[task_type] = {}, {}
            for ckpt, recs in group_records_by_checkpoint(results[key]).items():
                agg = aggregate_over_folds(recs, task_type)
                for entry in agg:
                    entry["checkpoint"] = ckpt
                aggregated[task_type].extend(agg)
                per_dataset[task_type][ckpt] = mean_over_models(agg)
                summary[task_type][ckpt] = summarise_per_task_type(per_dataset[task_type][ckpt])
                final_scores[task_type][ckpt] = compute_final_score(agg, weight_for_config)
        else:
            agg = aggregate_over_folds(results[key], task_type)
            aggregated[task_type] = agg
            per_dataset[task_type] = mean_over_models(agg)
            summary[task_type] = summarise_per_task_type(per_dataset[task_type])
            final_scores[task_type] = compute_final_score(agg, weight_for_config)

    output = {
        **results,
        "aggregated": aggregated,
        "per_dataset": per_dataset,
        "summary": summary,
        "final_scores": final_scores,
    }

    output_path = cfg.get("results_output_path", "results.yaml")
    with open(output_path, "w") as outfile:
        yaml.dump(output, outfile, sort_keys=False)

    print_collected_results(aggregated, summary, final_scores, multi_checkpoint)


# Single scalar score per task type, derived from a task's mean metrics. Edit this mapping
# to change what each task type contributes to the summary / final score.
def score_from_metric_means(task_type, metric_means):
    if task_type == "segmentation":
        return metric_means.get("dice")
    if task_type == "classification":
        vals = [v for v in (metric_means.get("Precision"), metric_means.get("Recall")) if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    if task_type == "regression":
        return metric_means.get("MAE")
    return None


def build_weight_map(cfg):
    """Map each box task's config name to its `weight` (default 1.0) for the final score."""
    weight_for_config = {}
    for group in ("segmentation_tasks", "classification_tasks", "regression_tasks"):
        for task in cfg.get(group) or []:
            if task is None:
                continue
            name = task.get("task")
            if name is None:
                continue
            weight_for_config[name] = float(task.get("weight", 1.0))
    return weight_for_config


PRIMARY_KEY = "primary"


def _mean_std_n(values):
    """Mean, sample std (ddof=1), and count for a list of numbers. std is 0.0 for n < 2,
    where the sample std is undefined."""
    mean = round(float(np.mean(values)), 4)
    std = round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(values)}


def aggregate_over_folds(raw_results_for_type, task_type):
    """collapse folds. Group raw per-fold records by (config, dataset) and reduce each
    metric to mean/std/n across that group's folds."""
    grouped = {}
    for inference_data in raw_results_for_type:
        for dataset, record in inference_data.items():
            key = (record.get("config"), dataset)
            metrics = record.get("metrics", {})
            primary = score_from_metric_means(task_type, metrics)
            grouped.setdefault(key, {})
            for metric, value in metrics.items():
                grouped[key].setdefault(metric, []).append(value)
            if primary is not None:
                grouped[key].setdefault(PRIMARY_KEY, []).append(primary)

    aggregated = []
    for (config, dataset), metric_values in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        metric_stats = {metric: _mean_std_n(values) for metric, values in metric_values.items()}
        aggregated.append(
            {
                "task": dataset,
                "config": config,
                "metrics": metric_stats,
                "score": metric_stats.get(PRIMARY_KEY, {}).get("mean"),
            }
        )
    return aggregated


def mean_over_models(aggregated_for_type):
    """collapse models. Regroup entries by dataset and average each metric's
    per-model means across all models of that dataset (std/n are now across models)."""
    grouped = {}
    for entry in aggregated_for_type:
        dataset = entry["task"]
        grouped.setdefault(dataset, {})
        for metric, stats in entry["metrics"].items():
            grouped[dataset].setdefault(metric, []).append(stats["mean"])

    per_dataset = {}
    for dataset, metric_means in sorted(grouped.items()):
        per_dataset[dataset] = {metric: _mean_std_n(values) for metric, values in metric_means.items()}
    return per_dataset


def summarise_per_task_type(per_dataset_for_type):
    """collapse datasets. Average each dataset's primary score across all datasets of
    the type (std/n across datasets). Returns None if no dataset has a primary score."""
    primary_means = [stats[PRIMARY_KEY]["mean"] for stats in per_dataset_for_type.values() if PRIMARY_KEY in stats]
    if not primary_means:
        return None
    summary = _mean_std_n(primary_means)
    return {"score": summary["mean"], "std": summary["std"], "n_tasks": summary["n"]}


def compute_final_score(aggregated_for_type, weight_for_config):
    """Weighted mean of per-task scores, weighting each task by its box-config `weight`."""
    numerator, denominator = 0.0, 0.0
    for a in aggregated_for_type:
        if a.get("score") is None:
            continue
        weight = weight_for_config.get(a["config"], 1.0)
        numerator += weight * a["score"]
        denominator += weight
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def group_records_by_checkpoint(raw_results_for_type):
    """Split the raw per-run records by their source checkpoint_run_id."""
    grouped = {}
    for inference_data in raw_results_for_type:
        ckpt = next(iter(inference_data.values()), {}).get("checkpoint_run_id")
        grouped.setdefault(ckpt, []).append(inference_data)
    return grouped


def _short_config(config):
    """Last path segment of a config name, for compact table rows."""
    return config.rsplit("/", 1)[-1] if config else str(config)


def print_collected_results(aggregated, summary, final_scores, multi_checkpoint=False):
    # Per-task table (raw metrics as mean±std over folds) for each task type. In all-models
    # mode a leading `checkpoint` column is added; single-checkpoint output is unchanged.
    for task_type in ("segmentation", "classification", "regression"):
        rows = aggregated.get(task_type) or []
        if not rows:
            continue
        metric_names = []
        for entry in rows:
            for metric in entry["metrics"]:
                if metric != PRIMARY_KEY and metric not in metric_names:
                    metric_names.append(metric)
        labels = {m: f"{m} (mean±std, n)" for m in metric_names}

        def cell(entry, metric):
            stats = entry["metrics"].get(metric)
            return f"{stats['mean']} ± {stats['std']} (n={stats['n']})" if stats else "-"

        lead_keys = (["checkpoint"] if multi_checkpoint else []) + ["task", "config"]

        def lead_val(entry, key):
            if key == "config":
                return _short_config(entry["config"])
            return str(entry.get(key))

        lead_w = {k: max([len(k)] + [len(lead_val(e, k)) for e in rows]) for k in lead_keys}
        col_w = {m: max(len(labels[m]), max(len(cell(e, m)) for e in rows)) for m in metric_names}

        lead_hdr = "  ".join(f"{k:<{lead_w[k]}}" for k in lead_keys)
        header = f"{lead_hdr}  " + "  ".join(f"{labels[m]:<{col_w[m]}}" for m in metric_names)
        print(f"\n=== {task_type}_results ===")
        print(header)
        print("-" * len(header))
        ordered = sorted(rows, key=lambda e: tuple(lead_val(e, k) for k in lead_keys)) if multi_checkpoint else rows
        for entry in ordered:
            lead = "  ".join(f"{lead_val(entry, k):<{lead_w[k]}}" for k in lead_keys)
            cells = "  ".join(f"{cell(entry, m):<{col_w[m]}}" for m in metric_names)
            print(f"{lead}  {cells}")

    # Headline summary: one line per task type, or per (task type, checkpoint) in all-models mode.
    print("\n=== EvalBox summary ===")
    for task_type in summary:
        summ = summary[task_type]
        if summ is None:
            continue
        if multi_checkpoint:
            for ckpt, stats in sorted(summ.items(), key=lambda kv: str(kv[0])):
                if stats is None:
                    continue
                final = (final_scores.get(task_type) or {}).get(ckpt)
                print(
                    f"{task_type:>14} [ckpt {ckpt}]: mean score {stats['score']} ± {stats['std']}"
                    f" over {stats['n_tasks']} task(s) | weighted final score {final}"
                )
        else:
            final = final_scores.get(task_type)
            print(
                f"{task_type:>14}: mean score {summ['score']} ± {summ['std']} over {summ['n_tasks']} task(s)"
                f" | weighted final score {final}"
            )


def resolve_subconfigs_for_config(config):
    seg_tasks, cls_tasks, regr_tasks = [], [], []
    for task in config.get("segmentation_tasks") or []:
        if task is not None:
            seg_tasks.append(task.get("task"))
    for task in config.get("classification_tasks") or []:
        if task is not None:
            cls_tasks.append(task.get("task"))
    for task in config.get("regression_tasks") or []:
        if task is not None:
            regr_tasks.append(task.get("task"))
    return seg_tasks, cls_tasks, regr_tasks


def _infer_task_type(job_name):
    """Map a Hydra job name (e.g. 'finetune_seg', 'train_cls') to a task type."""
    name = (job_name or "").lower()
    if "seg" in name:
        return "segmentation"
    if "cls" in name:
        return "classification"
    if "reg" in name:
        return "regression"
    return None


def _resolved_value(d, *keys, default=None):
    """Read nested keys from a plain dict, treating unresolved OmegaConf
    interpolations (e.g. '${task}') as missing."""
    value = d
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    if isinstance(value, str) and value.startswith("${"):
        return default
    return value


def get_run_metadata_from_hydra(run_dir):
    """Recover a run's metadata from the files Hydra writes for every run:
    ``hydra/config.yaml`` (resolved scalar config values) and ``hydra/hydra.yaml``
    (job name / config name and the resolved output dir). This avoids depending on the
    run-dir naming convention. ``run_id`` is the only path-derived field and is taken
    from Hydra's own resolved ``runtime.output_dir``. Returns None if files are missing.
    """
    config_path = os.path.join(run_dir, "hydra", "config.yaml")
    hydra_path = os.path.join(run_dir, "hydra", "hydra.yaml")
    if not (os.path.isfile(config_path) and os.path.isfile(hydra_path)):
        return None
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(hydra_path) as f:
        hydra_cfg = yaml.safe_load(f).get("hydra", {})

    output_dir = _resolved_value(hydra_cfg, "runtime", "output_dir", default=run_dir)
    run_id_match = re.search(r"run_id=(\d+)", output_dir or "")

    return {
        "config": _resolved_value(hydra_cfg, "job", "config_name"),
        "task": _resolved_value(cfg, "task"),
        "task_type": _infer_task_type(_resolved_value(hydra_cfg, "job", "name", default="")),
        "fold": _resolved_value(cfg, "data", "fold"),
        "runID": int(run_id_match.group(1)) if run_id_match else None,
        "train_split": _resolved_value(cfg, "data", "train_split"),
        "test_split": _resolved_value(cfg, "data", "test_split"),
        "checkpoint_run_id": _resolved_value(cfg, "checkpoint_run_id"),
        "load_checkpoint_name": _resolved_value(cfg, "load_checkpoint_name"),
        "root": _resolved_value(cfg, "root"),
    }


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
            raise ValueError(f"Unexpected task type: {task_type}")
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
