import torch


class Torch_ClampTarget:
    def __init__(self, clamp: bool = False, min_value: float = 0.0, max_value: float = 1.0):
        self.clamp = clamp
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data_dict: dict) -> dict:
        if self.clamp and "label" in data_dict:
            data_dict["label"] = torch.clamp(data_dict["label"], min=self.min_value, max=self.max_value)
        return data_dict
