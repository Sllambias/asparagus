import torch
from gardening_tools.modules.networks.components.blocks import MultiLayerConvDropoutNormNonlin
from gardening_tools.modules.networks.unet import UNet, UNetCLSREG


def unet_b_lw_dec(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    use_skip_connections: bool = True,
):
    return UNet(
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        use_skip_connections=use_skip_connections,
    )


def unet_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
):
    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        deep_supervision=deep_supervision,
    )


def unet_m(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
):
    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=64,
        deep_supervision=deep_supervision,
    )


def unet_clsreg_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
):
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        deep_supervision=deep_supervision,
    )


def unet_clsreg_s(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
):
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        starting_filters=16,
    )


def unet_tiny(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
):
    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=2,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        deep_supervision=deep_supervision,
    )


def unet_clsreg_tiny(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
):
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=2,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
    )


if __name__ == "__main__":
    model = unet_b_lw_dec(input_channels=1, output_channels=1)
    print(model)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print(y.shape)
