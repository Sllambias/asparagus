from typing import Literal
import torch
import torch.nn as nn
import logging
from yucca.modules.networks.networks.YuccaNet import YuccaNet
from asparagus.modules.networks.blocks.conv_blocks import (
    DoubleConvDropoutNormNonlin,
    MultiLayerConvDropoutNormNonlin,
)


class UNetEncoder(nn.Module):

    def __init__(
        self,
        input_channels: int,
        basic_block=DoubleConvDropoutNormNonlin,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        weightInitializer=None,
        pool_op=nn.MaxPool3d,
        starting_filters: int = 64,
    ) -> None:
        super().__init__()

        # Task specific
        self.filters = starting_filters

        # Model parameters
        self.basic_block = basic_block
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.pool_op = pool_op

        self.in_conv = self.basic_block(
            input_channels=input_channels,
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool1 = self.pool_op(2)
        self.encoder_conv1 = self.basic_block(
            input_channels=self.filters,
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool2 = self.pool_op(2)
        self.encoder_conv2 = self.basic_block(
            input_channels=self.filters * 2,
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool3 = self.pool_op(2)
        self.encoder_conv3 = self.basic_block(
            input_channels=self.filters * 4,
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool4 = self.pool_op(2)
        self.encoder_conv4 = self.basic_block(
            input_channels=self.filters * 8,
            output_channels=self.filters * 16,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.pool1(x0)
        x1 = self.encoder_conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.encoder_conv2(x2)

        x3 = self.pool3(x2)
        x3 = self.encoder_conv3(x3)

        x4 = self.pool4(x3)
        x4 = self.encoder_conv4(x4)

        return [x0, x1, x2, x3, x4]


class UNetDecoder(nn.Module):

    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 64,
        basic_block=DoubleConvDropoutNormNonlin,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        deep_supervision=False,
        dropout_in_decoder=False,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        upsample_op=torch.nn.ConvTranspose3d,
        use_skip_connections=True,
        weightInitializer=None,
    ) -> None:
        super().__init__()

        # Task specific
        self.num_classes = output_channels
        self.filters = starting_filters

        # Model parameters
        self.basic_block = basic_block
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.deep_supervision = deep_supervision
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.use_skip_connections = use_skip_connections
        self.upsample = upsample_op

        self.upsample1 = self.upsample(self.filters * 16, self.filters * 8, kernel_size=2, stride=2)
        self.decoder_conv1 = self.basic_block(
            input_channels=self.filters * (16 if self.use_skip_connections else 8),
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample2 = self.upsample(self.filters * 8, self.filters * 4, kernel_size=2, stride=2)
        self.decoder_conv2 = self.basic_block(
            input_channels=self.filters * (8 if self.use_skip_connections else 4),
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample3 = self.upsample(self.filters * 4, self.filters * 2, kernel_size=2, stride=2)
        self.decoder_conv3 = self.basic_block(
            input_channels=self.filters * (4 if self.use_skip_connections else 2),
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample4 = self.upsample(self.filters * 2, self.filters, kernel_size=2, stride=2)
        self.decoder_conv4 = self.basic_block(
            input_channels=self.filters * (2 if self.use_skip_connections else 1),
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.out_conv = self.conv_op(self.filters, self.num_classes, kernel_size=1)

        if self.deep_supervision:
            self.ds_out_conv0 = self.conv_op(self.filters * 16, self.num_classes, kernel_size=1)
            self.ds_out_conv1 = self.conv_op(self.filters * 8, self.num_classes, kernel_size=1)
            self.ds_out_conv2 = self.conv_op(self.filters * 4, self.num_classes, kernel_size=1)
            self.ds_out_conv3 = self.conv_op(self.filters * 2, self.num_classes, kernel_size=1)

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, xs):
        x_enc = xs[4]

        if self.use_skip_connections:
            x5 = torch.cat([self.upsample1(x_enc), xs[3]], dim=1)
            x5 = self.decoder_conv1(x5)

            x6 = torch.cat([self.upsample2(x5), xs[2]], dim=1)
            x6 = self.decoder_conv2(x6)

            x7 = torch.cat([self.upsample3(x6), xs[1]], dim=1)
            x7 = self.decoder_conv3(x7)

            x8 = torch.cat([self.upsample4(x7), xs[0]], dim=1)
            x8 = self.decoder_conv4(x8)
        else:
            x5 = self.decoder_conv1(self.upsample1(x_enc))
            x6 = self.decoder_conv2(self.upsample2(x5))
            x7 = self.decoder_conv3(self.upsample3(x6))
            x8 = self.decoder_conv4(self.upsample4(x7))

        # We only want to do multiple outputs during training, therefore it is only enabled
        # when grad is also enabled because that means we're training. And if for some reason
        # grad is enabled and you're not training, then there's other, bigger problems.
        if self.deep_supervision and torch.is_grad_enabled():
            ds0 = self.ds_out_conv0(xs[4])
            ds1 = self.ds_out_conv1(x5)
            ds2 = self.ds_out_conv2(x6)
            ds3 = self.ds_out_conv3(x7)
            ds4 = self.out_conv(x8)
            return [ds4, ds3, ds2, ds1, ds0]

        logits = self.out_conv(x8)

        return logits


class UNet(YuccaNet):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        encoder: nn.Module = UNetEncoder,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder: nn.Module = UNetDecoder,
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        dimensions: str = "3D",
        starting_filters: int = 32,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        if dimensions == "2D":
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
            pool_op = nn.MaxPool2d
            upsample_op = torch.nn.ConvTranspose2d
        elif dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
            upsample_op = torch.nn.ConvTranspose3d
        else:
            logging.warn("Uuh, dimensions not in ['2D', '3D']")

        self.encoder = encoder(
            basic_block=encoder_basic_block,
            conv_op=conv_op,
            dropout_op=dropout_op,
            input_channels=input_channels,
            norm_op=norm_op,
            pool_op=pool_op,
            starting_filters=starting_filters,
        )
        self.decoder = decoder(
            basic_block=decoder_basic_block,
            conv_op=conv_op,
            dropout_op=dropout_op,
            norm_op=norm_op,
            output_channels=output_channels,
            starting_filters=starting_filters,
            upsample_op=upsample_op,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, x):
        enc = self.encoder(x)
        return self.decoder(enc)


def unet_b_lw_dec(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
):

    return UNet(
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        use_skip_connections=False,
    )


def unet_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
):

    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
    )


if __name__ == "__main__":
    model = unet_b_lw_dec(input_channels=1, output_channels=1)
    print(model)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print(y.shape)
