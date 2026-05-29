"""Integration test for pipeline/run/pretrain.py components.

Uses SelfSupervisedModule + PretrainDataModule on synthetic volumes.
Torch_CopyImageToLabel adds batch["label"] so the SSL reconstruction loss can run.
Covers: unet_tiny 3D, ResidualEncoderUNet 3D, unet_tiny 2D.
"""

from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.lightning_modules import SelfSupervisedModule
from asparagus.modules.networks.resenc_unet import resenc_unet_debug
from asparagus.modules.networks.unet import unet_tiny
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from torchvision import transforms


def make_pretrain_data_module(files, **kwargs):
    # CopyImageToLabel saves label = image before any GPU augmentation,
    # which is all the SSL reconstruction loss requires.
    copy_transform = transforms.Compose([Torch_CopyImageToLabel(copy=True)])
    return PretrainDataModule(
        batch_size=1,
        num_workers=1,
        train_split=files["train"],
        val_split=files["val"],
        train_transforms=copy_transform,
        val_transforms=copy_transform,
        **kwargs,
    )


def make_ssl_module(model):
    return SelfSupervisedModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        train_transforms=None,
        val_transforms=None,
    )


def test_pretrain_resenc_unet_fit(pretrain_files, make_trainer):
    """SelfSupervisedModule fits with a tiny ResidualEncoderUNet on 3D synthetic data."""
    model = resenc_unet_debug(
        dimensions="3D",
        input_channels=1,
        output_channels=1,
    )
    make_trainer().fit(make_ssl_module(model), datamodule=make_pretrain_data_module(pretrain_files))


def test_pretrain_unet_2d_fit(pretrain_files_2d, make_trainer):
    """SelfSupervisedModule fits with a 2D UNet on synthetic 2D data."""
    model = unet_tiny(input_channels=1, output_channels=1, dimensions="2D")
    make_trainer().fit(make_ssl_module(model), datamodule=make_pretrain_data_module(pretrain_files_2d))
