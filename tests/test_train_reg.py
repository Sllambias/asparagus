"""Integration test for pipeline/run/train_reg.py components.

Uses RegressionModule + ClsRegDataModule + unet_clsreg_tiny on synthetic (image, label) data.
Runs both trainer.fit() and trainer.test() mirroring the full pipeline.
"""
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules import RegressionModule
from asparagus.modules.networks.unet import unet_clsreg_tiny


def test_train_reg_fit_and_test(reg_files, tmp_path, make_trainer):
    """RegressionModule fits then runs inference on synthetic data."""
    model = unet_clsreg_tiny(input_channels=1, output_channels=1, dimensions="3D")

    data_module = ClsRegDataModule(
        batch_size=1,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=reg_files["train"],
        val_split=reg_files["val"],
        test_samples=reg_files["test"],
        use_random_datasampler=False,
    )

    module = RegressionModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        test_output_path=str(tmp_path / "preds.json"),
    )

    trainer = make_trainer(limit_test_batches=1)
    trainer.fit(module, datamodule=data_module)
    trainer.test(module, datamodule=data_module)
