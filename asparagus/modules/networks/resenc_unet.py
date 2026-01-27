import logging
import torch
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.blocks import ResidualBlock
from gardening_tools.modules.networks.components.encoders import ResidualUNetEncoder
from gardening_tools.modules.networks.components.heads import ClsRegHead
from gardening_tools.modules.networks.resunet import ResidualEncoderUNet
from torch import nn
from typing import List, Tuple, Type, Union


class ResidualEncoderUNetCLSREG(BaseNet):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str,
        kernel_size: int,
        stride: int,
        features_per_stage: list,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = True,
        encoder_basic_block: Type[ResidualBlock] = ResidualBlock,
        decoder: nn.Module = ClsRegHead,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    ):
        super().__init__()

        # Extract dropout rates from kwargs
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        encoder_dropout_rate = dropout_op_kwargs.get("encoder_dropout_rate", 0.0)
        decoder_dropout_rate = dropout_op_kwargs.get("decoder_dropout_rate", 0.0)
        inplace = dropout_op_kwargs.get("inplace", True)

        if dimensions == "2D":
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
            pool_op = nn.MaxPool2d
            clsreg_pool_op = nn.AdaptiveAvgPool2d
        elif dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
            clsreg_pool_op = nn.AdaptiveAvgPool3d
        else:
            logging.warning("Uuh, dimensions not in ['2D', '3D']")

        self.num_classes = output_channels

        self.stem_weight_name = "encoder.stem.conv1.conv.weight"

        self.encoder = ResidualUNetEncoder(
            input_channels=input_channels,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_size=kernel_size,
            stride=stride,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs={"p": encoder_dropout_rate, "inplace": inplace},
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=encoder_basic_block,
            pool_op=pool_op,
        )

        self.decoder = decoder(
            pool_op=clsreg_pool_op,
            input_channels=features_per_stage[-1],
            output_channels=output_channels,
            dropout_rate=decoder_dropout_rate,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def forward_with_features(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output, skips[-1]


# Encoder 29M parameters
# Full model 42M parameters
# This is the "classic" unet, but with residual encoder blocks
def resenc_unet_s(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


# Encoder 90M parameters
# Full model 102M parameters
def resenc_unet_b(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    use_skip_connections=True,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
        use_skip_connections=use_skip_connections,
    )


# Encoder 90M parameters
# Full model - 90.3 M    Total params
def resenc_unet_b_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    dropout_op_kwargs: dict = None,
):
    return ResidualEncoderUNetCLSREG(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        dropout_op_kwargs=dropout_op_kwargs,
    )


# Encoder 345M parameters
# Full model 391M parameters
def resenc_unet_l(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 620, 620),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


# Encoder 602M parameters
# Full model 662M parameters
def resenc_unet_h(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 768, 768),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 8, 8),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


# Encoder 989M parameters
# Full model 1079M parameters
def resenc_unet_g(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 1024, 1024),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 8, 8),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )
