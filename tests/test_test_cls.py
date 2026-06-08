"""Integration test for pipeline/run/test_cls.py components.

Mirrors test_cls.py's two-stage flow:
  1. Train briefly, save a checkpoint.
  2. Load that checkpoint, run test-time inference on new data.
Covers: unet_clsreg_tiny 3D, ResidualEncoderUNetCLSREG 3D, PrimusCLSREG 3D, unet_clsreg_tiny 2D.

Note: batch_size=2 is required. ClassificationModule.on_before_batch_transfer
uses squeeze() on labels; with batch_size=1 this collapses [B] to 0-dim,
causing CrossEntropyLoss to fail with "batch_size (1) vs (0)".
"""

import pytest
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules import ClassificationModule
from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetCLSREG
from asparagus.modules.networks.unet import unet_clsreg_tiny
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict


def run_cls_two_stage(model_fn, files, tmp_path, make_trainer, ckpt_stem):
    """Train → checkpoint → load weights → test inference. Shared by all cls tests."""
    ckpt_path = tmp_path / f"{ckpt_stem}.ckpt"

    # --- Stage 1: train and save a checkpoint ---
    train_model = model_fn()
    train_module = ClassificationModule(
        model=train_model,
        learning_rate=1e-3,
        warmup_epochs=0,
        test_output_path=str(tmp_path / f"{ckpt_stem}_train_preds.json"),
    )
    train_dm = ClsRegDataModule(
        batch_size=1,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=files["train"],
        val_split=files["val"],
        use_random_datasampler=False,
    )
    train_trainer = make_trainer()
    train_trainer.fit(train_module, datamodule=train_dm)
    train_trainer.save_checkpoint(str(ckpt_path))

    # --- Stage 2: load weights and run inference (mirrors test_cls.py logic) ---
    weights = load_checkpoint_state_dict(str(ckpt_path))
    infer_model = model_fn()
    infer_module = ClassificationModule(
        model=infer_model,
        weights=weights,
        test_output_path=str(tmp_path / f"{ckpt_stem}_test_preds.json"),
    )
    test_dm = ClsRegDataModule(
        batch_size=1,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=None,
        val_split=None,
        test_samples=files["test"],
        use_random_datasampler=False,
    )
    make_trainer(limit_test_batches=2).test(infer_module, datamodule=test_dm)


def test_test_cls_inference(cls_probe_files, tmp_path, make_trainer):
    """ClassificationModule runs test-time inference from a saved checkpoint."""
    run_cls_two_stage(
        lambda: unet_clsreg_tiny(input_channels=1, output_channels=2, dimensions="3D"),
        cls_probe_files,
        tmp_path,
        make_trainer,
        "unet3d",
    )


def test_test_cls_resenc_unet_inference(cls_probe_files, tmp_path, make_trainer):
    """ClassificationModule checkpoint → inference with ResidualEncoderUNetCLSREG (3D)."""
    run_cls_two_stage(
        lambda: ResidualEncoderUNetCLSREG(
            input_channels=1,
            output_channels=2,
            dimensions="3D",
            features_per_stage=(4, 8),
            stride=2,
            kernel_size=3,
            n_blocks_per_stage=(1, 1),
        ),
        cls_probe_files,
        tmp_path,
        make_trainer,
        "resenc_unet3d",
    )


def test_test_cls_primus_inference(cls_probe_files, tmp_path, make_trainer):
    """ClassificationModule checkpoint → inference with PrimusCLSREG (3D, requires timm)."""
    pytest.importorskip("timm")
    from asparagus.modules.networks.primus import PrimusCLSREG

    run_cls_two_stage(
        lambda: PrimusCLSREG(
            input_channels=1,
            output_channels=2,
            embed_dim=24,
            patch_embed_size=(8, 8, 8),
            eva_depth=1,
            eva_numheads=2,
            input_shape=(32, 32, 32),
        ),
        cls_probe_files,
        tmp_path,
        make_trainer,
        "primus3d",
    )


def test_test_cls_unet_2d_inference(cls_probe_files_2d, tmp_path, make_trainer):
    """ClassificationModule checkpoint → inference with a 2D UNet."""
    run_cls_two_stage(
        lambda: unet_clsreg_tiny(input_channels=1, output_channels=2, dimensions="2D"),
        cls_probe_files_2d,
        tmp_path,
        make_trainer,
        "unet2d",
    )
