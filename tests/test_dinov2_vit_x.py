"""Integration and unit tests for DINOv2Module + vit_x.

Three tests cover:
  1. Pretraining smoke test  — DINOv2Module + PretrainDataModule + DINOv2Augmentation
  2. Classification smoke test — frozen vit_x backbone + linear head (linear-eval protocol)
  3. Forward-structure test  — verifies output keys/shapes match the official DinoV2 spec
"""

import lightning as L
import torch
import torch.nn as nn
from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules.dinov2 import DINOv2Module
from asparagus.modules.networks.dinov2 import vit_x
from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation


class DINOv2LinearHead(L.LightningModule):
    """Frozen vit_x backbone + trainable linear classification head.

    Mirrors the official DinoV2 linear-evaluation protocol: backbone weights
    are fixed; only the single linear layer is trained.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, hidden_size: int):
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch["CLSREG_label"] = batch["CLSREG_label"].squeeze(-1).long()
        return batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone.encode(x)  # (B, hidden_size)
        return self.head(feat)

    def training_step(self, batch, batch_idx):
        return self.loss_fn(self(batch["image"]), batch["CLSREG_label"])

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self.loss_fn(self(batch["image"]), batch["CLSREG_label"]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.head.parameters(), lr=0.1)


def test_dinov2_pretrain_vit_x(pretrain_files, make_trainer):
    """DINOv2Module fits on synthetic pretrain data using a tiny vit_x backbone.

    Uses img_size=(32,32,32) / patch_size=(8,8,8) → 4³=64 patches, which is
    small enough to run on CPU in ~seconds while exercising the full pipeline:
    DINOv2Augmentation → PretrainDataModule → DINOv2Module (DINO + iBOT + KoLeo
    losses, layer-wise LR, teacher EMA update).
    """
    model = vit_x(img_size=(32, 32, 32), patch_size=(8, 8, 8))

    # DINOv2Augmentation is used as a CPU transform: converts each raw image
    # dict into {"image": [global_view1, global_view2], "local_views": [lv1, lv2]}.
    # PyTorch's default collate then stacks per-sample lists into batched tensors.
    augmentation = DINOv2Augmentation(
        global_view_size=(32, 32, 32),
        global_view_scale=[0.5, 1.0],
        local_view_size=(32, 32, 32),
        local_view_scale=[0.1, 0.5],
        num_local_views=2,
    )

    data_module = PretrainDataModule(
        batch_size=1,
        num_workers=1,
        train_split=pretrain_files["train"],
        val_split=pretrain_files["val"],
        train_transforms=augmentation,
        val_transforms=None,
    )

    module = DINOv2Module(model=model, learning_rate=1e-3, teacher_temp_warmup_steps=0, warmup_epochs=0)

    # limit_val_batches=0: DINOv2Module.validation_step raises NotImplementedError
    # by design ("set val_steps_per_epoch to 0").
    make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)


def test_dinov2_cls_vit_x(cls_probe_files, make_trainer):
    """Frozen vit_x backbone + linear head classifies on synthetic cls data.

    vit_x.encode() extracts (B, 192) CLS-token features from 32³ volumes.
    Only the linear head is trained; backbone gradients are disabled.
    This mirrors the standard DinoV2 linear-evaluation protocol.
    """
    backbone = vit_x(img_size=(32, 32, 32), patch_size=(8, 8, 8))
    module = DINOv2LinearHead(backbone=backbone, num_classes=2, hidden_size=192)

    data_module = ClsRegDataModule(
        batch_size=2,
        num_workers=2,
        train_split=cls_probe_files["train"],
        val_split=cls_probe_files["val"],
        test_samples=cls_probe_files["test"],
        use_random_datasampler=False,
    )

    make_trainer().fit(module, datamodule=data_module)


def test_dinov2_vit_x_forward_structure():
    """vit_x output dict matches the official DinoV2 interface.

    Checks:
    - All expected keys are present with correct shapes.
    - Teacher outputs have no gradient (frozen backbone).
    - encode() returns (B, hidden_size) CLS tokens for downstream use.
    """
    model = vit_x(img_size=(16, 16, 16), patch_size=(8, 8, 8))
    model.eval()

    B = 2
    x = torch.randn(B, 1, 16, 16, 16)

    out = model(global_views=[x])["pred"]

    n_patches = (16 // 8) ** 3  # 2³ = 8
    projection_dim = 65536
    hidden_size = 192

    assert out["teacher_cls_token"].shape == (B, projection_dim)
    assert out["student_cls_token"].shape == (B, projection_dim)
    assert out["teacher_patch_tokens"].shape == (B, n_patches, projection_dim)
    assert out["student_patch_tokens"].shape == (B, n_patches, projection_dim)
    assert out["mask"].shape == (B, n_patches)
    assert not out["teacher_cls_token"].requires_grad  # teacher is frozen

    feat = model.encode(x)
    assert feat.shape == (B, hidden_size)
