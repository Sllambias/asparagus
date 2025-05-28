import torch
from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform


def torch_mask(image: torch.Tensor, pixel_value: float, ratio: float, token_size: list[int]) -> torch.Tensor:
    """
    input should be 4d tensor of shape (c, x, y, z) or 3d tensor of shape (c, x, y)
    """
    input_shape = image.shape[1:]
    if len(token_size) == 1:
        token_size *= len(input_shape)
    assert len(input_shape) == len(
        token_size
    ), f"mask token size not compatible with input data — token: {token_size}, image shape: {input_shape}"

    input_shape_tensor = image.new_tensor(input_shape, dtype=torch.int)
    token_size_tensor = image.new_tensor(token_size, dtype=torch.int)

    grid_dims = torch.ceil(input_shape_tensor / token_size_tensor).to(dtype=torch.int)
    grid_size = torch.prod(grid_dims).item()

    grid_flat = image.new_ones(grid_size)
    grid_flat[: int(grid_size * ratio)] = 0
    grid_flat = grid_flat[torch.randperm(grid_size, device=image.device)]

    grid = grid_flat.view(*grid_dims)

    for dim, size in enumerate(token_size):
        grid = grid.repeat_interleave(size, dim=dim)

    slices = tuple(slice(0, s) for s in input_shape)
    mask = grid[slices]

    image[:, mask == 0] = pixel_value

    return image, mask


class Torch_Mask(YuccaTransform):
    def __init__(
        self,
        data_key: str = "image",
        mask_key: str = "mask",
        pixel_value: float = 0.0,
        ratio: float = 0.6,
        token_size: list[int] = [4],
    ):
        self.data_key = data_key
        self.mask_key = mask_key
        self.pixel_value = pixel_value
        self.ratio = ratio
        self.token_size = token_size

    @staticmethod
    def get_params():
        pass

    def __mask__(
        self,
        image,
    ):
        image, mask = torch_mask(
            image=image,
            pixel_value=self.pixel_value,
            ratio=self.ratio,
            token_size=self.token_size,
        )
        return image, mask

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        data_dict[self.mask_key] = torch.zeros_like(data_dict[self.data_key])
        for b in range(data_dict[self.data_key].shape[0]):
            image, mask = self.__mask__(data_dict[self.data_key][b])
            data_dict[self.data_key][b] = image
            data_dict[self.mask_key][b] = mask
        return data_dict
