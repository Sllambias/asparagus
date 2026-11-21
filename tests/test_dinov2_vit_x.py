"""Integration and unit tests for DINOv2Module + vit_x.

Three tests cover:
  1. Pretraining smoke test  — DINOv2Module + PretrainDataModule + DINOv2Augmentation
  2. Classification smoke test — frozen vit_x backbone + linear head (linear-eval protocol)
  3. Forward-structure test  — verifies output keys/shapes match the official DinoV2 spec
"""

import datetime
import lightning as L
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules.dinov2 import DINOv2Module
from asparagus.modules.losses.ibot import IBOTPatchLoss3D
from asparagus.modules.networks.dinov2 import vit_x
from asparagus.modules.transforms.blockmask import DINOv2Collate
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


def test_2d_dinov2_pretrain_vit_x(pretrain_pngs_2d, make_trainer):
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
        train_split=pretrain_pngs_2d["train"],
        val_split=pretrain_pngs_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
        collate_fn=DINOv2Collate(global_view_size=(32, 32), patch_size=(8, 8)),
    )

    module = DINOv2Module(model=model, learning_rate=1e-3, teacher_temp_warmup_epochs=0, warmup_epochs=0)

    # limit_val_batches=0: DINOv2Module.validation_step raises NotImplementedError
    # by design ("set training.val_batches_per_epoch_per_device to 0").
    make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)


def test_3d_dinov2_pretrain_vit_x(pretrain_uint8_volumes, make_trainer):
    """DINOv2Module fits on synthetic 8-bit volumes using a tiny vit_x backbone (4^3 = 64 patches)."""
    model = vit_x(img_size=(32, 32, 32), patch_size=(8, 8, 8))
    augmentation = DINOv2Augmentation(
        global_view_size=(32, 32, 32),
        global_view_scale=[0.5, 1.0],
        local_view_size=(16, 16, 16),
        local_view_scale=[0.1, 0.5],
        num_local_views=2,
    )
    data_module = PretrainDataModule(
        batch_size=1,
        num_workers=1,
        train_split=pretrain_uint8_volumes["train"],
        val_split=pretrain_uint8_volumes["val"],
        train_transforms=augmentation,
        val_transforms=None,
        collate_fn=DINOv2Collate(global_view_size=(32, 32, 32), patch_size=(8, 8, 8)),
    )
    module = DINOv2Module(model=model, learning_rate=1e-3, teacher_temp_warmup_epochs=0, warmup_epochs=0)
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

    masks = DINOv2Collate(global_view_size=(16, 16, 16), patch_size=(8, 8, 8)).random_block_mask(B).flatten(1)
    out = model(global_views=[x], masks=masks)["pred"]

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


def test_2d_dinov2_pretrain_without_ibot(pretrain_pngs_2d, make_trainer):
    """With ibot_loss_weight=0 the patch heads are skipped and training still runs."""
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    out = model(global_views=[torch.randn(2, 1, 32, 32)] * 2, ibot=False)["pred"]
    assert "masked_student_global_patch_tokens_after_head" not in out

    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=2)
    data_module = PretrainDataModule(
        batch_size=4,
        num_workers=1,
        train_split=pretrain_pngs_2d["train"],
        val_split=pretrain_pngs_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
        collate_fn=DINOv2Collate(global_view_size=(32, 32), patch_size=(8, 8)),
    )
    module = DINOv2Module(model=model, teacher_temp_warmup_epochs=0, warmup_epochs=0, ibot_loss_weight=0.0)
    assert module.criterion.ibot_loss_fn is None
    make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)


def tiny_2d_setup(pretrain_pngs_2d, **module_kwargs):
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=2)
    data_module = PretrainDataModule(
        batch_size=4,
        num_workers=1,
        train_split=pretrain_pngs_2d["train"],
        val_split=pretrain_pngs_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
        collate_fn=DINOv2Collate(global_view_size=(32, 32), patch_size=(8, 8)),
    )
    module = DINOv2Module(model=model, teacher_temp_warmup_epochs=0, warmup_epochs=0, **module_kwargs)
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


@pytest.mark.parametrize("img_size, local_size", [((32, 32), (16, 16)), ((32, 32, 32), (16, 16, 16))])
def test_local_views_train_the_position_embeddings(img_size, local_size):
    model = vit_x(img_size=img_size, patch_size=(8,) * len(img_size), projection_dim=256)
    # A plain sum of the final LayerNorm output has zero gradient, so square it
    model.student_backbone(torch.randn(2, 1, *local_size))[:, 0].pow(2).sum().backward()
    grad = model.student_backbone.vit.patch_embedding.position_embeddings.grad
    assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("img_size", [(32, 32), (16, 16, 16)])
def test_masked_tokens_keep_their_position(img_size):
    model = vit_x(img_size=img_size, patch_size=(8,) * len(img_size), projection_dim=256)
    backbone = model.student_backbone
    mask = torch.zeros(1, backbone.sequence_length - 1, dtype=torch.bool)
    mask[0, :4] = True
    with torch.no_grad():
        masked_outputs = backbone(torch.randn(1, 1, *img_size), mask=mask)[0, 1:][mask[0]]
    # Without a position, all masked tokens of an image enter the transformer alike and come out alike
    assert not torch.allclose(masked_outputs[0], masked_outputs[1])


def test_koleo_uses_backbone_cls_tokens():
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), projection_dim=256)
    masks = DINOv2Collate(global_view_size=(32, 32), patch_size=(8, 8)).random_block_mask(4).flatten(1)
    out = model(global_views=[torch.randn(2, 1, 32, 32)] * 2, masks=masks)["pred"]
    assert out["student_global_cls_token_before_head"].shape == (4, 192)


def test_last_layer_is_frozen_during_the_first_epoch(pretrain_pngs_2d, make_trainer):
    for freeze_last_layer_epochs, expect_frozen in [(1, True), (0, False)]:
        model, data_module, module = tiny_2d_setup(pretrain_pngs_2d, freeze_last_layer_epochs=freeze_last_layer_epochs)
        last_layer = model.student_dino_head.last_layer.weight_v.detach().clone()
        first_layer = model.student_dino_head.layers[0].weight.detach().clone()
        make_trainer(limit_val_batches=0).fit(module, datamodule=data_module)
        assert torch.equal(last_layer, model.student_dino_head.last_layer.weight_v) == expect_frozen
        assert not torch.equal(first_layer, model.student_dino_head.layers[0].weight)


def test_teacher_updates_once_per_optimizer_step(pretrain_pngs_2d, make_trainer):
    model, data_module, module = tiny_2d_setup(pretrain_pngs_2d)
    steps = []
    update_teacher = model.update_teacher

    def counting_update_teacher(global_step, max_steps):
        steps.append(global_step)
        update_teacher(global_step=global_step, max_steps=max_steps)

    model.update_teacher = counting_update_teacher
    make_trainer(limit_val_batches=0, limit_train_batches=4, accumulate_grad_batches=2).fit(module, datamodule=data_module)
    assert steps == [1, 2]


def update_ibot_center_on_rank(rank: int, store_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{store_file}", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30)
    )
    # Rank 0 has no masked patches, rank 1 has three
    mask = torch.zeros(2, 4, dtype=torch.bool)
    if rank == 1:
        mask[0, :3] = True
    tokens = torch.ones(int(mask.sum()), 8)
    loss_fn = IBOTPatchLoss3D(output_dim=8, center_momentum=0.9)
    loss = loss_fn(tokens, tokens.clone(), mask)
    torch.save((loss, loss_fn.center.center), f"{store_file}_{rank}.pt")
    dist.destroy_process_group()


def test_ibot_center_syncs_when_a_rank_has_no_masked_patches(tmp_path):
    store_file = str(tmp_path / "store")
    torch.multiprocessing.spawn(update_ibot_center_on_rank, args=(store_file,), nprocs=2)
    for rank in range(2):
        loss, center = torch.load(f"{store_file}_{rank}.pt")
        assert torch.isfinite(loss)
        # Both ranks move to the mean token of rank 1: 0.9 * 0 + 0.1 * 1
        assert torch.allclose(center, torch.full((1, 8), 0.1)), rank


def test_teacher_temperature_warms_up_over_epochs(pretrain_pngs_2d, make_trainer):
    model, data_module, module = tiny_2d_setup(pretrain_pngs_2d)
    module.teacher_temp_warmup_epochs = 2
    make_trainer(limit_val_batches=0, limit_train_batches=4, accumulate_grad_batches=2).fit(module, datamodule=data_module)
    # 2 optimizer steps per epoch
    assert module.criterion.teacher_temp_warmup_steps == 4


OFFICIAL_BACKBONE = dict(
    num_layers=3,
    num_heads=4,
    mlp_dim=384,
    qkv_bias=True,
    layer_scale_init=1e-5,
    drop_path_rate=0.3,
    num_register_tokens=4,
)


@pytest.mark.parametrize("img_size", [(32, 32), (16, 16, 16)])
def test_default_backbone_runs_monai_blocks(img_size):
    model = vit_x(img_size=img_size, patch_size=(8,) * len(img_size), projection_dim=256)
    backbone = model.student_backbone.eval()
    x = torch.randn(2, 1, *img_size)
    vit = backbone.vit
    with torch.no_grad():
        tokens = vit.patch_embedding.patch_embeddings(x).flatten(2).transpose(1, 2) + vit.patch_embedding.position_embeddings
        tokens = torch.cat((vit.cls_token.expand(2, -1, -1), tokens), dim=1)
        for blk in vit.blocks:
            tokens = blk(tokens)
        assert torch.allclose(backbone(x), vit.norm(tokens), atol=1e-6)


@pytest.mark.parametrize("img_size", [(32, 32), (16, 16, 16)])
def test_official_backbone_options(img_size):
    model = vit_x(img_size=img_size, patch_size=(8,) * len(img_size), hidden_size=96, projection_dim=256, **OFFICIAL_BACKBONE)
    student, teacher = model.student_backbone, model.teacher_backbone
    for name, value in student.state_dict().items():
        assert torch.equal(value, teacher.state_dict()[name]), name
    # Drop path regularizes the student only, as official
    assert student.drop_path.p == 0.3 and teacher.drop_path.p == 0.0
    assert student.vit.blocks[0].attn.qkv.bias is not None
    assert torch.allclose(student.vit.blocks[2].ls2.gamma, torch.full((96,), 1e-5))
    # The registers are left out of the output: CLS token, then the patches
    assert student(torch.randn(2, 1, *img_size)).shape == (2, student.sequence_length, 96)


def test_layer_scale_is_decayed_per_block_without_weight_decay(pretrain_pngs_2d, make_trainer):
    model = vit_x(img_size=(32, 32), patch_size=(8, 8), hidden_size=96, projection_dim=256, **OFFICIAL_BACKBONE)
    augmentation = DINOv2Augmentation(global_view_size=(32, 32), local_view_size=(16, 16), num_local_views=2)
    data_module = PretrainDataModule(
        batch_size=4,
        num_workers=1,
        train_split=pretrain_pngs_2d["train"],
        val_split=pretrain_pngs_2d["val"],
        train_transforms=augmentation,
        val_transforms=None,
        collate_fn=DINOv2Collate(global_view_size=(32, 32), patch_size=(8, 8)),
        sample_with_replacement=False,
    )
    module = DINOv2Module(model=model, teacher_temp_warmup_epochs=0, warmup_epochs=0)
    trainer = make_trainer(limit_val_batches=0, limit_train_batches=2)
    trainer.fit(module, datamodule=data_module)
    groups = {group["name"]: group for group in trainer.optimizers[0].param_groups}
    gamma = groups["model.student_backbone.vit.blocks.1.ls1.gamma"]
    qkv = groups["model.student_backbone.vit.blocks.1.attn.qkv.weight"]
    assert gamma["lr"] == qkv["lr"] < groups["model.student_dino_head.layers.0.weight"]["lr"]
    assert gamma["weight_decay"] == 0.0 and qkv["weight_decay"] > 0.0
    assert groups["model.student_backbone.register_tokens"]["lr"] == groups["model.student_backbone.mask_token"]["lr"]
