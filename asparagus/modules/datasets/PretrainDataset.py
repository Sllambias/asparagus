import torch
import torchvision
from torch.utils.data import Dataset
from typing import Optional


class PretrainDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {"file_path": file, "image": data, "transforms_applied": {}}
        data_dict = self._transform(data_dict)  # CPU transforms only here

        if torch.isnan(data_dict["image"]).any() or torch.isinf(data_dict["image"]).any():
            # print(f"Case contains NaNs or infs: {file}")
            data_dict["image"] = torch.nan_to_num(data_dict["image"], nan=0.0, posinf=4.0, neginf=-1.0)

        if "label" in data_dict.keys() and data_dict["label"] is not None:
            if torch.isnan(data_dict["label"]).any() or torch.isinf(data_dict["label"]).any():
                # print(f"Case label contains NaNs or infs in label: {file}")
                data_dict["label"] = torch.nan_to_num(data_dict["label"], nan=0.0, posinf=4.0, neginf=-1.0)

        return data_dict

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict
