"""Integration test for pipeline/run/pretrain.py components.

Uses SelfSupervisedModule + PretrainDataModule + unet_tiny on synthetic 8x8x8 volumes.
Torch_CopyImageToLabel adds batch["label"] so the SSL reconstruction loss can run.
"""

from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.lightning_modules import SelfSupervisedModule
from asparagus.modules.networks.unet import unet_tiny
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from torchvision import transforms


def test_pretrain_fit(pretrain_files, make_trainer):
    """SelfSupervisedModule fits on synthetic pretrain data with reconstruction loss."""
    model = unet_tiny(input_channels=1, output_channels=1, dimensions="3D")

    # CopyImageToLabel saves label = image before any GPU augmentation,
    # which is all the SSL reconstruction loss requires.
    copy_transform = transforms.Compose([Torch_CopyImageToLabel(copy=True)])

    data_module = PretrainDataModule(
        batch_size=1,
        num_workers=1,
        train_split=pretrain_files["train"],
        val_split=pretrain_files["val"],
        train_transforms=copy_transform,
        val_transforms=copy_transform,
    )

    module = SelfSupervisedModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        train_transforms=None,
        val_transforms=None,
    )

    make_trainer().fit(module, datamodule=data_module)
