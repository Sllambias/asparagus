"""Integration test for pipeline/run/linear_probe.py components.

Mirrors linear_probe.py's three-phase flow: validate → fit → test.
LinearProbeModule calls model._encode() internally; only models that implement
_encode() are supported (ResidualEncoderUNetCLSREG, PrimusCLSREG — not UNetCLSREG).

batch_size=2 is required: squeeze(-1) in on_before_batch_transfer collapses
a [B] label tensor to 0-dim when B=1, causing CrossEntropyLoss to fail.
limit_test_batches=2 ensures both test files (labels 0 and 1) are processed
so MulticlassAUROC has both classes present.

Covers: ResidualEncoderUNetCLSREG 3D, PrimusCLSREG 3D, ResidualEncoderUNetCLSREG 2D.
"""

import pytest
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules import LinearProbeModule
from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetCLSREG


def run_linear_probe(model, files, tmp_path, make_trainer, dimensions, out_stem):
    """validate → fit → test. Shared by all linear-probe tests."""

    data_module = ClsRegDataModule(
        batch_size=1,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=files["train"],
        val_split=files["val"],
        test_samples=files["test"],
        use_random_datasampler=False,
    )
    module = LinearProbeModule(
        model=model,
        learning_rates=[0.1, 0.01],
        num_classes=2,
        dimensions=dimensions,
        test_output_path=str(tmp_path / f"{out_stem}_probe_preds.json"),
        weights=None,
    )
    trainer = make_trainer(limit_test_batches=2)
    data_module.setup("fit")
    trainer.validate(module, datamodule=data_module)
    trainer.fit(module, datamodule=data_module)
    trainer.test(module, datamodule=data_module)


def test_linear_probe_validate_fit_test(cls_probe_files, tmp_path, make_trainer):
    """LinearProbeModule runs all three phases: validate → fit → test."""
    model = ResidualEncoderUNetCLSREG(
        input_channels=1,
        output_channels=2,
        dimensions="3D",
        features_per_stage=(4, 8),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 1),
    )
    run_linear_probe(model, cls_probe_files, tmp_path, make_trainer, "3D", "resenc_unet3d")


def test_linear_probe_primus_validate_fit_test(cls_probe_files, tmp_path, make_trainer):
    """LinearProbeModule runs all three phases with PrimusCLSREG (3D, requires timm).
    PrimusCLSREG._encode() returns a tensor; LinearProbeModule.get_features() handles
    this via `deepest = skips[-1] if isinstance(skips, list) else skips`.
    """
    pytest.importorskip("timm")
    from asparagus.modules.networks.primus import PrimusCLSREG

    model = PrimusCLSREG(
        input_channels=1,
        output_channels=2,
        embed_dim=24,
        patch_embed_size=(8, 8, 8),
        eva_depth=1,
        eva_numheads=2,
        input_shape=(32, 32, 32),
    )
    run_linear_probe(model, cls_probe_files, tmp_path, make_trainer, "3D", "primus3d")


def test_linear_probe_resenc_unet_2d_validate_fit_test(cls_probe_files_2d, tmp_path, make_trainer):
    """LinearProbeModule runs all three phases with ResidualEncoderUNetCLSREG in 2D.
    LinearProbeModule(dimensions="2D") uses AdaptiveAvgPool2d in get_features().
    """
    model = ResidualEncoderUNetCLSREG(
        input_channels=1,
        output_channels=2,
        dimensions="2D",
        features_per_stage=(4, 8),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 1),
    )
    run_linear_probe(model, cls_probe_files_2d, tmp_path, make_trainer, "2D", "resenc_unet2d")
