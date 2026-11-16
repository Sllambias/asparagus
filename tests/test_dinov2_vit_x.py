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


def test_2d_dinov2_pretrain_vit_x(pretrain_files_2d, make_trainer):
    """DINOv2Module fits on synthetic pretrain data using a tiny vit_x backbone.

    Uses img_size=(32,32) / patch_size=(8,8) which is
    small enough to run on CPU in ~seconds while exercising the full pipeline:
    DINOv2Augmentation → PretrainDataModule → DINOv2Module (DINO + iBOT + KoLeo
    losses, layer-wise LR, teacher EMA update).
    """
    model = vit_x(img_size=(32, 32), patch_size=(8, 8))

    # DINOv2Augmentation is used as a CPU transform: converts each raw image
    # dict into {"image": [global_view1, global_view2], "local_views": [lv1, lv2]}.
    # PyTorch's default collate then stacks per-sample lists into batched tensors.
    augmentation = DINOv2Augmentation(
        global_view_size=(32, 32),
        global_view_scale=[0.5, 1.0],
        local_view_size=(32, 32),
        local_view_scale=[0.1, 0.5],
        num_local_views=4,
    )

    data_module = PretrainDataModule(
        batch_size=4,
        num_workers=1,
        train_split=pretrain_files_2d["train"],
        val_split=pretrain_files_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
    )

    module = DINOv2Module(model=model, learning_rate=1e-3, teacher_temp_warmup_steps=0, warmup_epochs=0)

    # limit_val_batches=0: DINOv2Module.validation_step raises NotImplementedError
    # by design ("set training.val_batches_per_epoch_per_device to 0").
    make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)


def test_3d_dinov2_pretrain_vit_x(pretrain_files, make_trainer):
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
    # by design ("set training.val_batches_per_epoch_per_device to 0").
    make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)


def test_3d_dinov2_cls_vit_x(cls_probe_files, make_trainer):
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


def test_3d_dinov2_vit_x_forward_structure():
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
    n_masked = int(out["mask"].sum())
    assert out["masked_teacher_patch_tokens_after_head"].shape == (n_masked, projection_dim)
    assert out["masked_student_global_patch_tokens_after_head"].shape == (n_masked, projection_dim)
    assert out["mask"].shape == (B, n_patches)
    assert not out["teacher_cls_token"].requires_grad  # teacher is frozen

    feat = model.encode(x)
    assert feat.shape == (B, hidden_size)


def test_2d_dinov2_pretrain_without_ibot(pretrain_files_2d, make_trainer):
    """With ibot_loss_weight=0 the patch heads are skipped and training still runs."""
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    out = model(global_views=[torch.randn(2, 1, 32, 32)] * 2, ibot=False)["pred"]
    assert "masked_student_global_patch_tokens_after_head" not in out

    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=2)
    data_module = PretrainDataModule(
        batch_size=4,
        num_workers=1,
        train_split=pretrain_files_2d["train"],
        val_split=pretrain_files_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
    )
    module = DINOv2Module(model=model, teacher_temp_warmup_steps=0, warmup_epochs=0, ibot_loss_weight=0.0)
    assert module.criterion.ibot_loss_fn is None
    make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)


def tiny_2d_setup(pretrain_files_2d, **module_kwargs):
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=2)
    data_module = PretrainDataModule(
        batch_size=4,
        num_workers=1,
        train_split=pretrain_files_2d["train"],
        val_split=pretrain_files_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
    )
    module = DINOv2Module(model=model, teacher_temp_warmup_steps=0, warmup_epochs=0, **module_kwargs)
    return model, data_module, module


def test_teacher_starts_as_copy_of_student():
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256, ibot_separate_head=True)
    for student, teacher in [
        (model.student_backbone, model.teacher_backbone),
        (model.student_dino_head, model.teacher_dino_head),
        (model.student_ibot_head, model.teacher_ibot_head),
    ]:
        for name, value in student.state_dict().items():
            assert torch.equal(value, teacher.state_dict()[name]), name
    assert model.student_ibot_head is not model.student_dino_head
    assert not any(p.requires_grad for p in model.teacher_ibot_head.parameters())


def test_local_views_train_the_position_embeddings():
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    # A plain sum of the final LayerNorm output has zero gradient, so square it
    model.student_backbone(torch.randn(2, 1, 16, 16))[:, 0].pow(2).sum().backward()
    grad = model.student_backbone.vit.patch_embedding.position_embeddings.grad
    assert grad is not None and grad.abs().sum() > 0


def test_koleo_uses_backbone_cls_tokens():
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    out = model(global_views=[torch.randn(2, 1, 32, 32)] * 2)["pred"]
    assert out["student_global_cls_token_before_head"].shape == (4, 192)


def test_last_layer_is_frozen_during_the_first_epoch(pretrain_files_2d, make_trainer):
    for freeze_last_layer_epochs, expect_frozen in [(1, True), (0, False)]:
        model, data_module, module = tiny_2d_setup(pretrain_files_2d, freeze_last_layer_epochs=freeze_last_layer_epochs)
        last_layer = model.student_dino_head.last_layer.weight_v.detach().clone()
        first_layer = model.student_dino_head.layers[0].weight.detach().clone()
        make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)
        assert torch.equal(last_layer, model.student_dino_head.last_layer.weight_v) == expect_frozen
        assert not torch.equal(first_layer, model.student_dino_head.layers[0].weight)


def test_teacher_updates_once_per_optimizer_step(pretrain_files_2d, make_trainer):
    model, data_module, module = tiny_2d_setup(pretrain_files_2d)
    steps = []
    update_teacher = model.update_teacher

    def counting_update_teacher(global_step, max_steps):
        steps.append(global_step)
        update_teacher(global_step=global_step, max_steps=max_steps)

    model.update_teacher = counting_update_teacher
    make_trainer(limit_val_batches=0, limit_train_batches=4, accumulate_grad_batches=2).fit(module, datamodule=data_module)
    assert steps == [1, 2]
