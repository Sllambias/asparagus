from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation


def dinov2(global_view_size, local_view_size, patch_size):
    # ps to not break
    return DINOv2Augmentation(
        global_view_scale=[0.3, 1.0],
        global_view_size=global_view_size,
        local_view_scale=[0.3, 1.0],
        local_view_size=local_view_size,
        num_local_views=4,
    )
