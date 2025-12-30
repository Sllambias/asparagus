from asparagus.functional.decorators import depends_on_timm

try:
    from gardening_tools.modules.networks.primus import Primus
except ImportError:
    print(
        "Primus could not be imported. To use this network the optional dependency timm must be installed. "
        "timm can be installed manually or with pip install asparagus[extras]"
    )


@depends_on_timm()
def primus_s(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=396,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=12,
        eva_numheads=6,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
    )
    return model


@depends_on_timm()
def primus_b(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=792,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=12,
        eva_numheads=12,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_m(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=864,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=16,
        eva_numheads=12,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_l(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=1056,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=24,
        eva_numheads=16,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_h(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=1248,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=32,
        eva_numheads=16,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


@depends_on_timm()
def primus_g(input_channels, output_channels, patch_size, patch_embed_size=(8, 8, 8), patch_drop_rate=0.0):
    model = Primus(
        input_channels=input_channels,
        embed_dim=1584,
        patch_embed_size=patch_embed_size,
        num_classes=output_channels,
        eva_depth=32,
        eva_numheads=24,
        input_shape=patch_size,
        drop_path_rate=0.0,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=patch_drop_rate,
    )
    return model


if __name__ == "__main__":
    net = primus_s(1, 1, (64, 64, 64))
