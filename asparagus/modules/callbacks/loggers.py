import os
from argparse import Namespace
from lightning.fabric.utilities.logger import _convert_params
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from time import localtime, strftime, time
from typing import Any, Dict, Union


class BaseLogger(Logger):
    def __init__(
        self,
        file_name: str,
        save_dir: str = "./",
    ):
        super().__init__()
        self._file_name = file_name
        self._root_dir = save_dir

        self.epoch_start_time = time()
        self.log_file = None
        self.previous_epoch = -1
        self.current_epoch = 0
        self.hparams: Dict[str, Any] = {}
        self.NAME_HPARAMS_FILE = "hparams.yaml"
        self.step_metrics = {}

        if self.log_file is None:
            self.create_logfile()

    @property
    def version(self):
        return None

    @property
    def name(self):
        return None
        # return self._file_name

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def log_dir(self):
        log_dir = self.root_dir
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @rank_zero_only
    def create_logfile(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            "train_seg.log",
            # self.name + ".log",
        )
        with open(self.log_file, "w") as f:
            f.write(f"Starting model training \n {'log file:':20} {self.log_file} \n")
            print(f"Starting model training \n {'log file:':20} {self.log_file} \n")

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        params = _convert_params(params)
        self.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: dict, step):
        if "epoch" not in metrics:
            self.step_metrics.update(metrics)
            return
        self.current_epoch = metrics["epoch"]
        t = strftime("%Y_%m_%d_%H_%M_%S", localtime())
        with open(self.log_file, "a+") as f:
            if self.current_epoch > self.previous_epoch:
                epoch_end_time = time()
                f.write(f"\n \n{t} {'Current Epoch:':20} {self.current_epoch} \n")
                print(f"\n \n{t} {'Current Epoch:':20} {self.current_epoch}")
                f.write(f"{t} {'Epoch Time:':20} {epoch_end_time - self.epoch_start_time} \n")
                print(f"{t} {'Epoch Time:':20} {epoch_end_time - self.epoch_start_time}")
                for key, val in self.step_metrics.items():
                    f.write(f"{t} {key + ':':20} {val} \n")
                    print(f"{t} {key + ':':20} {val}")
                self.previous_epoch = self.current_epoch
                self.epoch_start_time = epoch_end_time
                self.step_metrics = {}
            for key, val in metrics.items():
                if "_step" in key or key == "epoch":
                    continue
                f.write(f"{t} {key + ':':20} {metrics[key]} \n")
                print(f"{t} {key + ':':20} {metrics[key]}")

    @rank_zero_only
    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    @rank_zero_only
    def save(self) -> None:
        """Save recorded hparams into yaml."""
        super().save()
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

    @rank_zero_only
    def finalize(self, _status) -> None:
        self.save()
