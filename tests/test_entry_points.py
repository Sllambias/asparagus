"""Snapshot test of what each asp_* entry point builds from the default configs.

Every object a run script builds is replaced by a Recorder, so nothing trains. The test compares
what was built, with which arguments, and which methods were called, to tests/snapshots/<script>.json.
Run with UPDATE_SNAPSHOTS=1 to rewrite the snapshots after an intended change.
"""

import asparagus.functional.hydra
import importlib
import json
import os
import pytest
import sys
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path


class Recorder:
    """Stands in for an object a script builds and logs every call made on it."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __getattr__(self, attribute):
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        return Recorder(f"{self.name}.{attribute}", self.log)

    def __call__(self, *args, **kwargs):
        self.log["calls"].append({"call": self.name, "args": list(args), "kwargs": kwargs})
        return Recorder(f"{self.name}()", self.log)


def make_builder(name, log):
    """A stand-in for a class or factory the script calls directly."""

    def build(*args, **kwargs):
        if name == "ModelCheckpoint":
            built_name = f"ModelCheckpoint[{kwargs['filename']}]"
        else:
            built_name = name
        log["built"].setdefault(built_name, []).append({"args": list(args), "kwargs": kwargs})
        return Recorder(built_name, log)

    return build


def make_instantiate(log):
    """A stand-in for hydra's instantiate that names each object after its config key."""

    def instantiate(config, _partial_):
        name = config._get_full_key(None)
        log["configs"][name] = OmegaConf.to_container(config, resolve=True)
        return make_builder(name, log)

    return instantiate


def describe(value, tmp_root):
    """Turn a recorded value into plain JSON with the temporary directory replaced by <tmp>."""
    if isinstance(value, Recorder):
        return f"<{value.name}>"
    if isinstance(value, (DictConfig, ListConfig)):
        return describe(OmegaConf.to_container(value, resolve=True), tmp_root)
    if isinstance(value, dict):
        return {str(key): describe(item, tmp_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item, tmp_root) for item in value]
    if isinstance(value, str):
        return value.replace(tmp_root, "<tmp>")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def write_fake_task(data_dir):
    """A task folder with the split files the default configs point at. No image files are needed."""
    task_dir = data_dir / "EntryPointTask"
    task_dir.mkdir(parents=True)
    # Different values in the two sections, so the snapshots show which one each script reads.
    dataset_json = {
        "dataset_config": {"n_classes": 2, "n_modalities": 1},
        "metadata": {"n_classes": 3, "n_modalities": 4},
    }
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json))
    for split in ("split_75_15_10", "split_40_10_50", "split_99_01_00"):
        (task_dir / f"{split}.json").write_text(json.dumps([{"train": ["train_0.pt"], "val": ["val_0.pt"]}]))
    for split in ("TEST_75_15_10", "TEST_40_10_50"):
        (task_dir / f"{split}.json").write_text(json.dumps(["test_0.pt"]))


@pytest.fixture
def entry_point_env(tmp_path, monkeypatch):
    """Point every ASPARAGUS_* variable at the temporary directory so a local .env is never used."""
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    extra_configs_dir = tmp_path / "extra_configs"
    write_fake_task(tmp_path / "data")
    (tmp_path / "cwd").mkdir()

    monkeypatch.setenv("ASPARAGUS_CONFIGS", str(configs_dir))
    monkeypatch.setenv("ASPARAGUS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ASPARAGUS_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("ASPARAGUS_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("ASPARAGUS_RAW_LABELS", str(tmp_path / "labels"))
    for variable in (
        "ASPARAGUS_PRETRAIN_CONFIGS",
        "ASPARAGUS_TRAIN_CONFIGS",
        "ASPARAGUS_FINETUNE_CONFIGS",
        "ASPARAGUS_EVAL_BOX_CONFIGS",
    ):
        monkeypatch.setenv(variable, str(extra_configs_dir))
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    monkeypatch.chdir(tmp_path / "cwd")

    # Imported here because versioning reads ASPARAGUS_MODELS when it is first imported.
    from asparagus.functional import versioning
    from asparagus.pipeline.auto_configuration import checkpoint, versioning as pathing_module

    def detect_id_in_tmp(id):
        return versioning.detect_id(id, model_dir=str(tmp_path / "models"))

    monkeypatch.setattr(checkpoint, "detect_id", detect_id_in_tmp)
    monkeypatch.setattr(pathing_module, "detect_id", detect_id_in_tmp)
    return tmp_path


def run_script(script, overrides, monkeypatch):
    """Import a run script fresh, swap what it builds for recorders, and run its Hydra main."""
    for resolver in ("random", "version", "eval"):
        if OmegaConf.has_resolver(resolver):
            OmegaConf.clear_resolver(resolver)
    sys.modules.pop(f"asparagus.pipeline.run.{script}", None)
    module = importlib.import_module(f"asparagus.pipeline.run.{script}")

    log = {"configs": {}, "built": {}, "calls": []}
    monkeypatch.setattr(asparagus.functional.hydra, "instantiate", make_instantiate(log))
    for name in (
        "logging",
        "ModelCheckpoint",
        "TQDMProgressBar",
        "LearningRateMonitor",
        "LinearProbeModule",
        "load_checkpoint_state_dict",
        "CPU_clsreg_val_test_transforms_crop",
        "CPU_seg_test_transforms",
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, make_builder(name, log))

    GlobalHydra.instance().clear()
    monkeypatch.setattr(sys, "argv", [script, *overrides])
    module.main()
    return log


def check_snapshot(script, log, tmp_root):
    snapshot = describe(log, tmp_root)
    path = Path(__file__).parent / "snapshots" / f"{script}.json"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    expected = json.loads(path.read_text())
    assert snapshot == expected, (
        f"{script} builds something different from {path.name}. "
        "Run pytest with -vv for the full diff, or with UPDATE_SNAPSHOTS=1 if the change is intended."
    )


def training_overrides(*extra):
    return ["task=EntryPointTask", "+model=unet_tiny", "training.seed=0", "run_id=1001", *extra]


@pytest.mark.parametrize(
    ("script", "extra_overrides"),
    [
        ("pretrain", []),
        ("train_cls", []),
        ("train_reg", []),
        ("train_seg", []),
        # The finetune and linear probe defaults leave the test split to the task config.
        ("finetune_cls", ["data.test_split=TEST_75_15_10"]),
        ("finetune_reg", ["data.test_split=TEST_75_15_10"]),
        ("finetune_seg", ["data.test_split=TEST_40_10_50"]),
        ("linear_probe", ["data.test_split=TEST_75_15_10"]),
    ],
)
def test_training_entry_point(script, extra_overrides, entry_point_env, monkeypatch):
    log = run_script(script, training_overrides(*extra_overrides), monkeypatch)
    check_snapshot(script, log, str(entry_point_env))


@pytest.mark.parametrize(("script", "trained_by"), [("test_cls", "train_cls"), ("test_seg", "train_seg")])
def test_inference_entry_point(script, trained_by, entry_point_env, monkeypatch):
    run_script(trained_by, training_overrides(), monkeypatch)
    log = run_script(script, ["test_task=EntryPointTask", "checkpoint_run_id=1001"], monkeypatch)
    check_snapshot(script, log, str(entry_point_env))
