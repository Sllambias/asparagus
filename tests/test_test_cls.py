"""Integration test for pipeline/run/test_cls.py components.

Mirrors test_cls.py's two-stage flow:
  1. Train briefly, save a checkpoint.
  2. Load that checkpoint, run test-time inference on new data.
Uses ClassificationModule + ClsRegDataModule + unet_clsreg_tiny.

Note: batch_size=2 is required. ClassificationModule.on_before_batch_transfer
uses squeeze() on labels; with batch_size=1 this collapses [B] to 0-dim,
causing CrossEntropyLoss to fail with "batch_size (1) vs (0)".
"""
from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules import ClassificationModule
from asparagus.modules.networks.unet import unet_clsreg_tiny
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict


def test_test_cls_inference(cls_probe_files, tmp_path, make_trainer):
    """ClassificationModule runs test-time inference from a saved checkpoint."""
    ckpt_path = tmp_path / "cls_checkpoint.ckpt"

    # --- Stage 1: train and save a checkpoint ---
    train_model = unet_clsreg_tiny(input_channels=1, output_channels=2, dimensions="3D")
    train_module = ClassificationModule(
        model=train_model,
        learning_rate=1e-3,
        warmup_epochs=0,
        test_output_path=str(tmp_path / "train_preds.json"),
    )
    train_dm = ClsRegDataModule(
        batch_size=2,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=cls_probe_files["train"],
        val_split=cls_probe_files["val"],
        use_random_datasampler=False,
    )
    train_trainer = make_trainer()
    train_trainer.fit(train_module, datamodule=train_dm)
    train_trainer.save_checkpoint(str(ckpt_path))

    # --- Stage 2: load weights and run inference (mirrors test_cls.py logic) ---
    weights = load_checkpoint_state_dict(str(ckpt_path))
    infer_model = unet_clsreg_tiny(input_channels=1, output_channels=2, dimensions="3D")
    infer_module = ClassificationModule(
        model=infer_model,
        weights=weights,
        test_output_path=str(tmp_path / "test_preds.json"),
    )
    test_dm = ClsRegDataModule(
        batch_size=2,
        num_workers=2,  # val_dataloader uses num_workers//2; needs >=2
        train_split=None,
        val_split=None,
        test_samples=cls_probe_files["test"],
        use_random_datasampler=False,
    )
    make_trainer(limit_test_batches=2).test(infer_module, datamodule=test_dm)
