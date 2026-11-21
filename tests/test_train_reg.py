"""Integration test for pipeline/run/train_reg.py components.

Uses RegressionModule + ClsRegDataModule on synthetic (image, label) data.
Runs both trainer.fit() and trainer.test() mirroring the full pipeline.
Covers: unet_clsreg_tiny 3D, ResidualEncoderUNetCLSREG 3D, PrimusCLSREG 3D, unet_clsreg_tiny 2D.
"""

import os
import pytest
import torch
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules import RegressionModule
from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetCLSREG
from asparagus.modules.networks.unet import unet_clsreg_tiny
from hydra import compose, initialize_config_dir


def run_reg_fit_and_test(model, files, tmp_path, make_trainer, out_stem):
    data_module = ClsRegDataModule(
        batch_size=1,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=files["train"],
        val_split=files["val"],
        test_samples=files["test"],
        use_random_datasampler=False,
    )
    module = RegressionModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        test_output_path=str(tmp_path / f"{out_stem}_preds.json"),
    )
    trainer = make_trainer(limit_test_batches=1)
    trainer.fit(module, datamodule=data_module)
    trainer.test(module, datamodule=data_module)


def test_train_reg_resenc_unet_fit_and_test(reg_files, tmp_path, make_trainer):
    """RegressionModule fits then tests with ResidualEncoderUNetCLSREG (3D)."""
    model = ResidualEncoderUNetCLSREG(
        input_channels=1,
        output_channels=1,
        dimensions="3D",
        features_per_stage=(4, 8),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 1),
    )
    run_reg_fit_and_test(model, reg_files, tmp_path, make_trainer, "resenc_unet3d")


def test_train_reg_primus_fit_and_test(reg_files, tmp_path, make_trainer):
    """RegressionModule fits then tests with PrimusCLSREG (3D, requires timm)."""
    pytest.importorskip("timm")
    from asparagus.modules.networks.primus import PrimusCLSREG

    model = PrimusCLSREG(
        input_channels=1,
        output_channels=1,
        embed_dim=24,
        patch_embed_size=(8, 8, 8),
        eva_depth=1,
        eva_numheads=2,
        input_shape=(32, 32, 32),
    )
    run_reg_fit_and_test(model, reg_files, tmp_path, make_trainer, "primus3d")


def test_train_reg_unet_2d_fit_and_test(reg_files_2d, tmp_path, make_trainer):
    """RegressionModule fits then tests with a 2D UNet."""
    model = unet_clsreg_tiny(input_channels=1, output_channels=1, dimensions="2D")
    run_reg_fit_and_test(model, reg_files_2d, tmp_path, make_trainer, "unet2d")


def test_primus_clsreg_early_fusion_runs_transformer():
    """PrimusCLSREG sends early-fused input through the Eva transformer, not just the patch embedding."""
    pytest.importorskip("timm")
    from asparagus.modules.networks.primus import PrimusCLSREG

    model = PrimusCLSREG(
        input_channels=2,
        output_channels=1,
        embed_dim=24,
        patch_embed_size=(8, 8, 8),
        eva_depth=1,
        eva_numheads=2,
        input_shape=(32, 32, 32),
    )
    model(torch.randn(1, 2, 32, 32, 32)).sum().backward()
    assert all(param.grad is not None for param in model.eva.parameters())


@pytest.mark.parametrize("config_name", ["default_train_cls", "default_train_reg"])
def test_train_config_starts_new_run(config_name):
    """run_id reads the top-level resume_training, so cls/reg training must turn it off there."""
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name)
    assert cfg.resume_training is False
