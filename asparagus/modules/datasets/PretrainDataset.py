import torchvision
from asparagus.functional.loading import load_image_file
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
        data = load_image_file(file)
        data_dict = {"file_path": file, "image": data, "transforms_applied": {}}
        data_dict = self._transform(data_dict)  # CPU transforms only here
        return data_dict

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict
