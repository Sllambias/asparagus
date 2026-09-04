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
from asparagus.functional.visualization import create_SSL_visualizations, log_images_to_logger
from asparagus.modules.lightning_modules.base_module import BaseModule
from asparagus.modules.losses.ssl_reconstruction_wrapper import ReconstructionLoss
from torchvision import transforms
from typing import Optional


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

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        mask = batch.get("mask", None)
        pred, encoder_features = self.model.forward_with_features(x)

        loss = self.rec_loss_fn(pred, y, mask)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        if self.current_epoch == self.log_images_every_n_epoch and batch_idx == 0 and self.trainer.is_global_zero:
            with torch.no_grad():
                images, error_images = create_SSL_visualizations(x, y, pred, mask, self.current_epoch)
                log_images_to_logger(
                    self.trainer.loggers,
                    images,
                    step=self.global_step,
                    prefix="images/train",
                )
                log_images_to_logger(
                    self.trainer.loggers,
                    error_images,
                    step=self.global_step,
                    prefix="images/train_error",
                )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        mask = batch.get("mask", None)

        pred, encoder_features = self.model.forward_with_features(x)
        loss = self.rec_loss_fn(pred, y, mask)

        self.log(
            "val/loss",
            loss,
            on_step=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        # rank zero only
        if self.trainer.is_global_zero:
            images, error_images = create_SSL_visualizations(x, y, pred, mask, self.current_epoch)
            log_images_to_logger(self.trainer.loggers, images, step=self.global_step, prefix="images/val")
            log_images_to_logger(
                self.trainer.loggers,
                error_images,
                step=self.global_step,
                prefix="images/val_error_map",
            )

    def format_metrics(self, stage, metric_groups):
        """
        Format metrics with hierarchical naming: stage/module/metric on wandb and stage_module/metric on MLflow.
        """
        #
        metric_separator = "_" if self.mlflow_logging else "/"  # mlflow only supports one / (sigh)
        metrics = {}
        for module_name, metric_dict in metric_groups.items():
            for key, value in metric_dict.items():
                metrics[f"{stage}{metric_separator}{module_name}/{key}"] = value
        return metrics

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        embeddings = self.model.encoder(x)[-1]
        return embeddings


class SelfSupervisedModuleExtendedLogging(SelfSupervisedModule):
    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input")
            return None

        mask = batch.get("mask", None)
        pred, encoder_features = self.model.forward_with_features(x)

        loss = self.rec_loss_fn(pred, y, mask)
        assert not torch.isnan(loss), "Reconstruction loss is NaN"

        # Logging
        with torch.no_grad():
            metrics = {}
            transforms_applied = batch.get("transforms_applied", None)
            if self.global_step % self.log_every_n_steps == 0:  # dont compute if not being logged...
                metrics = {
                    "loss": loss_metrics.compute_train(loss, pred, y, mask, self.rec_loss_fn),
                    "features": feat_metrics.compute_train(encoder_features),
                    "masking": masking.compute(mask, x),
                    "performance": perf_metrics.compute(transforms_applied, x.shape[0]),
                    "stability": stability_metrics.compute_nan_inf_metrics(loss=loss, pred=pred, activations=encoder_features),
                }
                self.log_dict(
                    self.format_metrics("train", metrics),
                    sync_dist=True,
                    batch_size=self.trainer.datamodule.batch_size,
                )

            if self.current_epoch == self.log_images_every_n_epoch and batch_idx == 0 and self.trainer.is_global_zero:
                images, error_images = create_SSL_visualizations(x, y, pred, mask, self.current_epoch)
                log_images_to_logger(
                    self.trainer.loggers,
                    images,
                    step=self.global_step,
                    prefix="images/train",
                )
                log_images_to_logger(
                    self.trainer.loggers,
                    error_images,
                    step=self.global_step,
                    prefix="images/train_error",
                )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input")
            return None

        mask = batch.get("mask", None)

        pred, encoder_features = self.model.forward_with_features(x)
        loss = self.rec_loss_fn(pred, y, mask)
        assert not torch.isnan(loss), "Reconstruction loss is NaN"

        # Logging
        metrics = {
            "loss": loss_metrics.compute_val(loss, pred, y, mask, self.rec_loss_fn),
            "features": feat_metrics.compute_val(encoder_features, self.model),
            "distribution": dist_metrics.compute(x, pred, y, encoder_features),
            "reconstruction": recon_metrics.compute(pred, y, mask),
        }
        self.log_dict(
            self.format_metrics("val", metrics),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        # rank zero only
        if self.trainer.is_global_zero:
            images, error_images = create_SSL_visualizations(x, y, pred, mask, self.current_epoch)
            log_images_to_logger(self.trainer.loggers, images, step=self.global_step, prefix="images/val")
            log_images_to_logger(
                self.trainer.loggers,
                error_images,
                step=self.global_step,
                prefix="images/val_error_map",
            )

    def on_after_backward(self):
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
