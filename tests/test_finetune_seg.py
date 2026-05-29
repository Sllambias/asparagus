"""Integration test for pipeline/run/finetune_seg.py components.

Uses SegmentationModule + SegDataModule + unet_tiny on synthetic 8x8x8 seg volumes.
Only tests trainer.fit() — trainer.test() is excluded because SegTestDataset._get_src_label()
loads from ASPARAGUS_RAW_LABELS which is unavailable in CI.
"""

from asparagus.modules.data_modules.training import SegDataModule
from asparagus.modules.lightning_modules import SegmentationModule
from asparagus.modules.networks.unet import unet_tiny


def test_finetune_seg_fit(seg_files, make_trainer):
    """SegmentationModule fits from scratch (weights=None) on synthetic seg data."""
    model = unet_tiny(input_channels=1, output_channels=2, dimensions="3D")

    data_module = SegDataModule(
        batch_size=1,
        num_workers=1,
        train_split=seg_files["train"],
        val_split=seg_files["val"],
        train_transforms=None,
        val_transforms=None,
    )

    module = SegmentationModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        weights=None,
        inference_patch_size=[32, 32, 32],
    )

    make_trainer().fit(module, datamodule=data_module)
