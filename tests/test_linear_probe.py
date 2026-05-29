"""Integration test for pipeline/run/linear_probe.py components.

Mirrors linear_probe.py's three-phase flow: validate → fit → test.
Uses LinearProbeModule + ClsRegDataModule + ResidualEncoderUNetCLSREG (tiny).

LinearProbeModule calls model._encode() internally, which is defined on
ResidualEncoderUNetCLSREG but not on UNetCLSREG.

batch_size=2 is required: squeeze(-1) in on_before_batch_transfer collapses
a [B] label tensor to 0-dim when B=1, causing CrossEntropyLoss to fail.
limit_test_batches=2 ensures both test files (labels 0 and 1) are processed
so MulticlassAUROC has both classes present.
"""

from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules import LinearProbeModule
from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetCLSREG


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
    data_module = ClsRegDataModule(
        batch_size=1,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=cls_probe_files["train"],
        val_split=cls_probe_files["val"],
        test_samples=cls_probe_files["test"],
        use_random_datasampler=False,
    )

    module = LinearProbeModule(
        model=model,
        learning_rates=[0.1, 0.01],
        num_classes=2,
        dimensions="3D",
        test_output_path=str(tmp_path / "probe_preds.json"),
        weights=None,
    )

    trainer = make_trainer(limit_test_batches=2)
    data_module.setup("fit")
    trainer.validate(module, datamodule=data_module)
    trainer.fit(module, datamodule=data_module)
    trainer.test(module, datamodule=data_module)
