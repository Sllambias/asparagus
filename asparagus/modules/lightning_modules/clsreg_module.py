import logging
import os
import torch
import torch.nn as nn
import wandb
from abc import abstractmethod
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.modules.lightning_modules.base_module import BaseModule
from gardening_tools.functional.paths.write import save_json
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchvision import transforms
from typing import Optional


class ClsRegBase(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-2,
        warmup_epochs: int = 10,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: str = None,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        log_image_every_n_epochs: int = 50,
        test_output_path: str = None,
        repeat_stem_weights: bool = True,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            decoder_warmup_epochs=decoder_warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            weights=weights,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            repeat_stem_weights=repeat_stem_weights,
        )
        self.loss = None
        self.task_type = None
        self.num_classes = model.num_classes
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.test_output_path = test_output_path
        self.ignore_index_in_metrics = -1
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_test_metrics()

    @abstractmethod
    def configure_test_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def configure_metrics(self, prefix: str):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]

        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.trainer.datamodule.batch_size
        )

        metrics = self.train_metrics(pred, y)
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="train",
                task_type=self.task_type,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]

        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.trainer.datamodule.batch_size)

        metrics = self.val_metrics(pred, y)
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="val",
                task_type=self.task_type,
            )

    def on_test_epoch_start(self):
        self.results = {}
        self.predictions = []
        self.labels = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        outputs = self.model.forward(x)
        return outputs

    def on_test_epoch_end(self):
        avg_results = {}
        avg_results = self.test_metrics(
            torch.tensor(self.predictions, device=self.predictions[0].device),
            torch.tensor(self.labels, device=self.predictions[0].device),
        )
        avg_results = {key: value.cpu().numpy().tolist() for key, value in avg_results.items()}
        self.results["metrics"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)
        logging.info(f"Aggregated test results for {len(self.results)} files: {avg_results}")


class ClassificationModule(ClsRegBase):
    def __init__(self, label_smoothing: float = 0.0, loss_weight: list = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor(loss_weight) if loss_weight else None, label_smoothing=label_smoothing
        )
        self.task_type = "classification"

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "Precision": MulticlassPrecision(num_classes=self.num_classes, average=None),
                "Recall": MulticlassRecall(num_classes=self.num_classes, average=None),
            }
        )

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/acc": MulticlassAccuracy(num_classes=self.num_classes, average=None),
                f"{prefix}/auroc": MulticlassAUROC(num_classes=self.num_classes, average=None),
            },
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch["CLSREG_label"] = batch["CLSREG_label"].squeeze().long()
        return batch

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prediction = outputs.argmax(1).long()
        label = batch["CLSREG_label"]
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
        }
        self.predictions.append(prediction)
        self.labels.append(label)


class RegressionModule(ClsRegBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.MSELoss()
        self.task_type = "regression"

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/MSE": MeanSquaredError(num_outputs=self.num_classes),
            },
        )

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "MSE": MeanSquaredError(num_outputs=self.num_classes),
                "MAE": MeanAbsoluteError(num_outputs=self.num_classes),
            },
        )

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prediction = outputs.squeeze().float()
        label = batch["CLSREG_label"].squeeze().float()
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
        }
        self.predictions.append(prediction)
        self.labels.append(label)
