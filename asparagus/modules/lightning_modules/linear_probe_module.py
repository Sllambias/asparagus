import logging
import os
import torch
import torch.nn as nn
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.modules.lightning_modules.base_module import BaseModule
from gardening_tools.functional.paths.write import save_json
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
)
from torchvision import transforms
from typing import List, Optional


class LinearProbeModule(BaseModule):
    """DINOv2-style linear probing module.

    Trains multiple linear heads simultaneously (one per candidate LR) on frozen backbone features.
    Selects the best head by validation accuracy, then tests only that head.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rates: List[float],
        num_classes: int,
        dimensions: str = "3D",
        loss_weight: list = None,
        train_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        test_output_path: str = None,
        weights: dict = None,
        pretrained_target_size=None,
        target_size=None,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0,
    ):

        super().__init__(
            model=model,
            weights=weights,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            repeat_stem_weights=False,
            load_decoder=False,
            pretrained_target_size=pretrained_target_size,
            target_size=target_size,
        )

        self.save_hyperparameters(ignore=["model", "train_transforms", "val_transforms", "weights"])
        self.learning_rates = learning_rates
        self.num_classes = num_classes
        self.test_output_path = test_output_path
        self.optimizer_momentum = optimizer_momentum
        self.optimizer_weight_decay = optimizer_weight_decay

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        if dimensions == "3D":
            self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = self.model.decoder.fc.in_features

        self.heads = nn.ModuleDict()
        for lr in learning_rates:
            head_name = self._lr_to_linear_head_name(lr)
            head = self._make_head(feature_dim, num_classes)
            self.heads[head_name] = head

        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(loss_weight) if loss_weight else None)

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")

        # Test metrics (only for best head)
        self.test_metrics = self.configure_test_metrics()

        self.best_head_lr = None
        self.ignore_index_in_metrics = -1

    def _make_head(self, feature_dim: int, num_classes: int) -> nn.Module:
        head = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(head.bias)
        return head

    @staticmethod
    def _lr_to_linear_head_name(lr: float) -> str:
        return f"lr_{lr:.0e}".replace(".", "_").replace("+", "").replace("-", "m")

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            skips = self.model._encode(x)

        # only use the features from the encoder's last layer
        deepest = skips[-1] if isinstance(skips, list) else skips
        features = self.global_pool(deepest)
        return torch.flatten(features, 1)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch["CLSREG_label"] = batch["CLSREG_label"].squeeze(-1).long()
        return batch

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]
        features = self._get_features(x)

        total_loss = 0.0
        for head_name, head in self.heads.items():
            logits = head(features)
            loss = self.loss_fn(logits, y)
            total_loss = total_loss + loss
            self.log(
                f"train/{head_name}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
            self.train_metrics[head_name].update(logits.detach().float(), y.detach())
        return total_loss

    @torch.no_grad()
    def on_train_epoch_end(self):
        for lr in self.learning_rates:
            head_name = self._lr_to_linear_head_name(lr)
            metrics = self.train_metrics[head_name].compute()
            formatted = format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics)
            self.log_dict(formatted, sync_dist=True)
            self.train_metrics[head_name].reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]
        features = self._get_features(x)

        for head_name, head in self.heads.items():
            logits = head(features)
            loss = self.loss_fn(logits, y)
            self.log(
                f"val/{head_name}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )

            self.val_metrics[head_name].update(logits.float(), y)

    def on_validation_epoch_end(self):
        current_aurocs = {}
        for lr in self.learning_rates:
            head_name = self._lr_to_linear_head_name(lr)
            metrics = self.val_metrics[head_name].compute()
            formatted = format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics)
            self.log_dict(formatted, sync_dist=True)

            current_aurocs[lr] = metrics[f"val/{head_name}/auroc_macro"].item()
            self.val_metrics[head_name].reset()

        best_lr, best_auroc = max(current_aurocs.items(), key=lambda x: x[1])
        self.best_head_lr = best_lr
        self.log("val/best_head_auroc", best_auroc, sync_dist=True)
        logging.info(f"Best head: lr={self.best_head_lr} with val auroc={best_auroc:.4f}")

    def configure_optimizers(self):
        param_groups = []
        for lr, (head_name, head) in zip(self.learning_rates, self.heads.items()):
            param_groups.append({"params": head.parameters(), "lr": lr, "name": head_name})

        optimizer = SGD(param_groups, momentum=self.optimizer_momentum, weight_decay=self.optimizer_weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "AUROC_macro": MulticlassAUROC(num_classes=self.num_classes, average="macro"),
                "AUPRC_macro": MulticlassAveragePrecision(num_classes=self.num_classes, average="macro"),
            }
        )

    def configure_metrics(self, prefix: str):
        metrics = nn.ModuleDict()
        for lr in self.learning_rates:
            head_name = self._lr_to_linear_head_name(lr)
            metrics[head_name] = MetricCollection(
                {
                    f"{prefix}/{head_name}/auroc_macro": MulticlassAUROC(num_classes=self.num_classes, average="macro"),
                    f"{prefix}/{head_name}/auprc_macro": MulticlassAveragePrecision(
                        num_classes=self.num_classes, average="macro"
                    ),
                }
            )
        return metrics

    def on_test_epoch_start(self):
        logging.info(f"Testing with head: {self._lr_to_linear_head_name(self.best_head_lr)} (lr={self.best_head_lr})")
        self.results = {}
        self.logits = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        features = self._get_features(x)
        logits = self.heads[self._lr_to_linear_head_name(self.best_head_lr)](features)

        label = batch["CLSREG_label"]
        self.results[batch["file_path"]] = {
            "prediction": logits.argmax(1).item(),
            "label": label.item(),
        }
        self.logits.append(logits.squeeze(0))
        self.labels.append(label)

    def on_test_epoch_end(self):
        logits_tensor = torch.stack(self.logits).float()
        labels_tensor = torch.stack(self.labels)

        avg_results = self.test_metrics(logits_tensor, labels_tensor)
        avg_results = {key: value.cpu().numpy().tolist() for key, value in avg_results.items()}

        self.results["metrics"] = avg_results
        self.results["best_head"] = self._lr_to_linear_head_name(self.best_head_lr)
        self.results["best_head_lr"] = self.best_head_lr
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)
        logging.info(f"Test using best head: {self._lr_to_linear_head_name(self.best_head_lr)} (lr={self.best_head_lr})")
        logging.info(f"Aggregated test results for {len(self.results)} files: {avg_results}")
