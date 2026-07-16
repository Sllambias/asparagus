"""
PyTorch LightningModule for 3D DINOv2 self-supervised training.
Handles optimizer, scheduler, training/validation steps, and teacher-student updates.
"""

import lightning as L
import math
import re
from asparagus.modules.losses.dinov2 import DINOv2Loss
from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer


class DINOv2Module(L.LightningModule):
    """
    PyTorch LightningModule for 3D DINOv2 self-supervised learning.
    Implements training, prediction, optimizer config, and teacher-student logic.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.0005,  # Reduced from 0.004 as per issue #6
        min_lr: float = 1e-6,
        weight_decay: float = 0.04,
        layer_decay: float = 0.9,
        gradient_clip_val: float = 3.0,
        teacher_temp_warmup_steps: int = 30,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        freeze_last_layer_epochs: int = 0,
        warmup_epochs: int = 10,
        compile_mode: bool = False,  # not currently used
        train_transforms: nn.Module = None,  # not currently used
        val_transforms: nn.Module = None,  # not currently used
        test_transforms: nn.Module = None,  # not currently used
        rec_loss_masked_only: bool = False,  # not currently used
        optimizer: str = "AdamW",
        mlflow_logging=False,  # not currently used
        log_every_n_steps=50,  # not currently used
        log_images_every_n_epoch=1,  # not currently used
    ) -> None:
        """
        Initialize the DINOv2Trainer3D LightningModule.
        Args: see model config for details.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_transforms", "val_transforms", "test_transforms"])
        self.lr = learning_rate
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.gradient_clip_val = gradient_clip_val
        self.optimizer = optimizer
        self.teacher_temp_warmup_steps = teacher_temp_warmup_steps
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        self.metrics = {"train": None, "val": None}
        self.warmup_epochs = warmup_epochs
        self.model = model

        # Loss
        self.criterion = DINOv2Loss(
            teacher_temp_min=teacher_temp_min,
            teacher_temp_max=teacher_temp_max,
            teacher_temp_warmup_steps=teacher_temp_warmup_steps,
            output_dim=self.model.projection_dim,
            ibot_loss_weight=1.0,
            koleo_loss_weight=0.1,
        )

    def predict_step(self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int) -> Tensor:
        raise NotImplementedError(
            "Predict step not implemented for DINOv2. Use encode() method directly for feature extraction."
        )

    def training_step(self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int) -> Tensor:
        outputs = self.model(global_views=batch["image"], local_views=batch.get("local_views", None))

        loss_dict = self.criterion(outputs["pred"], global_step=self.trainer.global_step)

        log_dict = {}
        for key, value in loss_dict.items():
            log_dict[f"train_{key}"] = value if value is not None else 0.0

        log_dict["global_step"] = float(self.trainer.global_step)
        self.log_dict(log_dict, prog_bar=False, sync_dist=True, batch_size=self.trainer.datamodule.batch_size)
        return loss_dict["total_loss"]

    def validation_step(self, batch: tuple[Tensor, Tensor, list[str]], batch_idx: int) -> Tensor:
        raise NotImplementedError("Validation not support for DinoV2 training. Set val_steps_per_epoch to 0.")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        # Calculate learning rate based on batch size
        lr_scale = math.sqrt(self.trainer.datamodule.batch_size * self.trainer.world_size / 1024)
        lr = self.lr * lr_scale
        num_layers = len(self.model.student_backbone.vit.blocks)

        def lr_layer(layer_idx: int) -> float:
            return self.layer_decay ** (num_layers + 1 - layer_idx)

        # Create parameter groups with layer-wise learning rates
        param_groups = []

        # Fix: Only include student parameters that require gradients
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Skip teacher parameters (they should not require gradients anyway)
            if "teacher" in name:
                continue

            # Skip if not student parameters
            if "student" not in name:
                continue

            group = {
                "name": name,
                "params": [param],
                "lr": lr,
                "weight_decay": self.weight_decay,
            }

            # Update lr based on layer
            if any(
                s in name
                for s in [
                    "pos_embed",
                    "mask_token",
                    "cls_token",
                    "register_tokens",
                ]
            ):
                group["lr"] = lr * lr_layer(0)
            elif "patch_embed" in name:
                group["lr"] = lr * lr_layer(0) * 0.2
            elif "residual" in name:
                group["lr"] = lr
            elif "blocks" in name:
                # Fix: More robust regex matching
                match = re.search(r"blocks\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    group["lr"] = lr * lr_layer(layer_idx + 1)
            elif "norm" in name:
                # Use default lr for norm layers
                pass
            elif "head" in name or "_dino_head" in name or "_ibot_head" in name:
                # Use default lr for heads
                pass
            else:
                # For any other student parameters, use default lr
                pass

            # Update weight_decay
            if name.endswith(".bias") or ".norm" in name or "gamma" in name:
                group["weight_decay"] = 0.0

            # Include parameter group
            param_groups.append(group)

        # Ensure we have parameters to optimize
        if not param_groups:
            raise ValueError("No student parameters found for optimization!")

        print(f"Found {len(param_groups)} parameter groups for optimization")
        print(f"Total parameters: {sum(len(group['params']) for group in param_groups)}")

        if self.optimizer == "AdamW":
            optimizer = AdamW(
                param_groups,
                lr=lr,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Fix: Ensure proper scheduler configuration
        max_steps = max(self.trainer.estimated_stepping_batches, 1)

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=max_steps,
                end_value=self.min_lr / lr,
            ),
            "interval": "step",
        }

        self.criterion.max_steps = max_steps
        self.criterion.max_epochs = self.trainer.max_epochs
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        # Gradient clipping as per issue #10
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        # Cancel last layer gradients during warmup (issue #5)
        # self.model.cancel_last_layer_gradients(self.current_epoch)

        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=max(self.trainer.estimated_stepping_batches, 1),
            start_value=0.04,
            end_value=0.4,
        )
        updates = []
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                updates.append({"name": group["name"], "weight_decay": weight_decay})

        update_param_groups(optimizer, updates=updates)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update of teacher - DDP handles synchronization automatically
        max_steps = max(self.trainer.estimated_stepping_batches, 1000)
        self.model.update_teacher(global_step=self.trainer.global_step, max_steps=max_steps)
        return super().on_train_batch_end(outputs, batch, batch_idx)
