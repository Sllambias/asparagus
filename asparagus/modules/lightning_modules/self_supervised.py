import logging
import torch
import torch.nn as nn
from asparagus.functional.metrics import (
    distribution as dist_metrics,
    features as feat_metrics,
    loss as loss_metrics,
    masking as masking,
    performance as perf_metrics,
    reconstruction as recon_metrics,
    stability as stability_metrics,
)
from asparagus.functional.visualization import create_ssl_visualizations, log_images_to_logger
from asparagus.modules.lightning_modules.base_module import BaseModule
from asparagus.modules.losses.ssl_reconstruction_wrapper import ReconstructionLoss
from torchvision import transforms
from typing import Any, Dict, List, Optional


class SelfSupervisedModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        log_images_every_n_epoch: int = 5,
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        rec_loss_masked_only: bool = False,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        optimizer: str = "AdamW",
        mlflow_logging: bool = False,
        log_every_n_steps: int = 50,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        weights: dict = None,
    ):
        super().__init__(
            model=model,
            warmup_epochs=warmup_epochs,
            learning_rate=learning_rate,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            weights=weights,
        )

        self.model = model
        self.rec_loss_fn = ReconstructionLoss(nn.MSELoss(reduction="mean"), rec_loss_masked_only=rec_loss_masked_only)
        self.log_images_every_n_epoch = log_images_every_n_epoch
        self.mlflow_logging = mlflow_logging
        self.log_every_n_steps = log_every_n_steps

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input")
            return None

        mask = batch.get("mask", None)
        pred, encoder_features = self.model.forward_with_features(x)
        loss = self.rec_loss_fn(pred, y, mask)
        assert not torch.isnan(loss), "Reconstruction loss is NaN"

        self.log_train_metrics(batch, pred, encoder_features, loss)
        if batch_idx == 0 and self.current_epoch % self.log_images_every_n_epoch == 0:
            self.log_images(x, y, pred, mask, stage="train")

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input")
            return

        mask = batch.get("mask", None)
        pred, encoder_features = self.model.forward_with_features(x)
        loss = self.rec_loss_fn(pred, y, mask)
        assert not torch.isnan(loss), "Reconstruction loss is NaN"

        self.log_val_metrics(batch, pred, encoder_features, loss)
        if batch_idx == 0:
            self.log_images(x, y, pred, mask, stage="val")

    def log_train_metrics(
        self, batch: Dict[str, Any], pred: torch.Tensor, encoder_features: List[torch.Tensor], loss: torch.Tensor
    ) -> None:
        self.log("train/loss", loss, on_step=True, sync_dist=True, batch_size=self.trainer.datamodule.batch_size)

    def log_val_metrics(
        self, batch: Dict[str, Any], pred: torch.Tensor, encoder_features: List[torch.Tensor], loss: torch.Tensor
    ) -> None:
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

    @torch.no_grad()
    def log_images(
        self, x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor], stage: str
    ) -> None:
        if not self.trainer.is_global_zero:
            return
        images, error_images = create_ssl_visualizations(x, y, pred, mask, self.current_epoch)
        log_images_to_logger(self.trainer.loggers, images, step=self.global_step, prefix=f"images/{stage}")
        log_images_to_logger(self.trainer.loggers, error_images, step=self.global_step, prefix=f"images/{stage}_error")

    def format_metrics(self, stage: str, metric_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format metrics with hierarchical naming: stage/module/metric on wandb and stage_module/metric on MLflow.
        """
        metric_separator = "_" if self.mlflow_logging else "/"  # mlflow only supports one / (sigh)
        metrics = {}
        for module_name, metric_dict in metric_groups.items():
            for key, value in metric_dict.items():
                metrics[f"{stage}{metric_separator}{module_name}/{key}"] = value
        return metrics

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        embeddings = self.model.encoder(x)[-1]
        return embeddings


class SelfSupervisedModuleExtendedLogging(SelfSupervisedModule):
    """SelfSupervisedModule that also logs feature, masking, distribution and stability metrics."""

    def log_train_metrics(
        self, batch: Dict[str, Any], pred: torch.Tensor, encoder_features: List[torch.Tensor], loss: torch.Tensor
    ) -> None:
        super().log_train_metrics(batch, pred, encoder_features, loss)
        if self.global_step % self.log_every_n_steps != 0:  # dont compute if not being logged...
            return

        x, y, mask = batch["image"], batch["label"], batch.get("mask", None)
        with torch.no_grad():
            metrics = {
                "loss": loss_metrics.compute_train(loss, pred, y, mask, self.rec_loss_fn.loss_fn),
                "features": feat_metrics.compute_train(encoder_features),
                "masking": masking.compute(mask, x),
                "performance": perf_metrics.compute(batch.get("transforms_applied", None), x.shape[0]),
                "stability": stability_metrics.compute_nan_inf_metrics(loss=loss, pred=pred, activations=encoder_features),
            }
        self.log_dict(
            self.format_metrics("train", metrics),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

    def log_val_metrics(
        self, batch: Dict[str, Any], pred: torch.Tensor, encoder_features: List[torch.Tensor], loss: torch.Tensor
    ) -> None:
        super().log_val_metrics(batch, pred, encoder_features, loss)

        x, y, mask = batch["image"], batch["label"], batch.get("mask", None)
        metrics = {
            "loss": loss_metrics.compute_val(loss, pred, y, mask, self.rec_loss_fn.loss_fn),
            "features": feat_metrics.compute_val(encoder_features, self.model),
            "distribution": dist_metrics.compute(x, pred, y, encoder_features),
            "reconstruction": recon_metrics.compute(pred, y, mask),
        }
        self.log_dict(
            self.format_metrics("val", metrics),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

    def on_after_backward(self) -> None:
        grad_clip_val = self.trainer.gradient_clip_val if hasattr(self.trainer, "gradient_clip_val") else None
        metrics_grouped = {
            "stability": stability_metrics.compute_on_backward(self.model, grad_clip_val),
            "performance": perf_metrics.compute_on_backward(self.trainer),
        }
        self.log_dict(
            self.format_metrics("train", metrics_grouped),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
