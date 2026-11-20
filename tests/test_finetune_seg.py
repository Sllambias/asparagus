"""Integration test for pipeline/run/finetune_seg.py components.

Uses SegmentationModule + SegDataModule on synthetic seg volumes.
Only tests trainer.fit() — trainer.test() is excluded because SegTestDataset._get_src_label()
loads from ASPARAGUS_RAW_LABELS which is unavailable in CI.
Covers: unet_tiny 3D, ResidualEncoderUNet 3D, unet_tiny 2D.
"""

from asparagus.modules.data_modules.training import SegDataModule
from asparagus.modules.lightning_modules import SegmentationModule
from asparagus.modules.networks.primus import primus_debug
from asparagus.modules.networks.resenc_unet import resenc_unet_debug
from asparagus.modules.networks.unet import unet_tiny
from asparagus.modules.transforms.presets import CPU_seg_train_transforms, GPU_all_train_transforms
from gardening_tools.modules.transforms.spatial import Torch_Spatial


def make_seg_data_module(files):
    return SegDataModule(
        batch_size=1,
        num_workers=1,
        train_split=files["train"],
        val_split=files["val"],
        train_transforms=None,
        val_transforms=None,
    )


def test_finetune_seg_primus_fit(seg_files, make_trainer):
    """SegmentationModule fits with a tiny ResidualEncoderUNet on 3D synthetic data."""
    model = primus_debug(
        input_channels=1,
        output_channels=2,
        input_shape=[32, 32, 32],
    )
    module = SegmentationModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        weights=None,
        inference_patch_size=[32, 32, 32],
    )
    make_trainer().fit(module, datamodule=make_seg_data_module(seg_files))


def test_finetune_seg_resenc_unet_fit(seg_files, make_trainer):
    """SegmentationModule fits with a tiny ResidualEncoderUNet on 3D synthetic data."""
    transforms = GPU_all_train_transforms(
        ndim=3,
        deep_supervision=True,
    )
    model = resenc_unet_debug(
        dimensions="3D",
        input_channels=1,
        output_channels=2,
        deep_supervision=True,
    )
    module = SegmentationModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        weights=None,
        inference_patch_size=[32, 32, 32],
        deep_supervision=True,
        train_transforms=transforms,
    )
    make_trainer().fit(module, datamodule=make_seg_data_module(seg_files))


def test_finetune_seg_unet_2d_fit(seg_files_2d, make_trainer):
    """SegmentationModule fits with a 2D UNet on synthetic 2D seg data."""
    model = unet_tiny(input_channels=1, output_channels=2, dimensions="2D")
    module = SegmentationModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        weights=None,
        inference_patch_size=[32, 32],
    )
    make_trainer().fit(module, datamodule=make_seg_data_module(seg_files_2d))


def test_seg_train_transforms_scale_probability():
    """Seg training scales a fifth of the patches, like the cls/reg presets."""
    transforms = CPU_seg_train_transforms([32, 32, 32]).transforms
    spatial = next(transform for transform in transforms if isinstance(transform, Torch_Spatial))
    assert spatial.p_scale_all_channel == 0.2

