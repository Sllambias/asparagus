"""Masked autoencoder pretraining of a 2D ResEnc-UNet on ultrasound PNGs and finetuning its encoder to classify."""

import torch
from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.lightning_modules import ClassificationModule, SelfSupervisedModule
from asparagus.modules.networks.resenc_unet import resenc_unet_b, resenc_unet_b_clsreg, resenc_unet_debug
from asparagus.modules.transforms.presets import (
    pretrain_GPU_train_transforms,
    pretrain_GPU_val_transforms,
    ultradino_mae_train,
    ultradino_mae_val,
)


def test_mae_pretraining_on_pngs(pretrain_pngs_2d, make_trainer):
    data_module = PretrainDataModule(
        batch_size=2,
        num_workers=1,
        train_split=pretrain_pngs_2d["train"],
        val_split=pretrain_pngs_2d["val"],
        train_transforms=ultradino_mae_train(patch_size=(32, 32)),
        val_transforms=ultradino_mae_val(patch_size=(32, 32)),
    )
    module = SelfSupervisedModule(
        model=resenc_unet_debug(dimensions="2D", input_channels=1, output_channels=1),
        learning_rate=1e-3,
        warmup_epochs=0,
        train_transforms=pretrain_GPU_train_transforms(masking=True, ndim=2),
        val_transforms=pretrain_GPU_val_transforms(masking=True),
    )
    make_trainer().fit(module, datamodule=data_module)


def test_the_classifier_loads_the_pretrained_encoder():
    """The networks of projects/DF/pretrain/ResEncUNetB_2D.yaml and finetune/ResEncUNetB_2D_preterm_P1_img_v7.yaml."""
    pretrained = resenc_unet_b(dimensions="2D", input_channels=1, output_channels=1, use_skip_connections=False)
    weights = SelfSupervisedModule(model=pretrained).state_dict()
    classifier = resenc_unet_b_clsreg(dimensions="2D", input_channels=1, output_channels=2)
    module = ClassificationModule(model=classifier, weights=weights, load_decoder=False)

    encoder = {name: param for name, param in module.state_dict().items() if name.startswith("model.encoder.")}
    assert len(encoder) > 0
    for name, param in encoder.items():
        assert torch.equal(param, weights[name]), name
