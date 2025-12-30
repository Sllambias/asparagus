from gardening_tools.modules.networks.resunet import ResidualEncoderUNet


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
