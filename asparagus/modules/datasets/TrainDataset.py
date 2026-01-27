import torch
import torchvision
from asparagus.paths import get_data_path, get_source_labels_path
from gardening_tools.functional.nibabel_utils import reorient_nib_image
from gardening_tools.functional.paths.read import load_pickle, read_file_to_nifti_or_np
from gardening_tools.functional.type_conversions import nifti_or_np_to_np
from torch.utils.data import Dataset
from typing import Optional


class SegDataset(Dataset):
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
        foreground_locations = load_pickle(file.replace(".pt", ".pkl"))["foreground_locations"]
        data_dict = {
            "file_path": file,
            "image": data[:-1],
            "label": data[-1:],
            "foreground_locations": foreground_locations,
            "transforms_applied": {},
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        data_dict.pop("foreground_locations")
        return data_dict


class ClsRegDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.composed_transforms = transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data[0],
            "CLSREG_label": data[1],
            "transforms_applied": {},
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict


class SegTestDataset(Dataset):
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
        properties = load_pickle(file.replace(".pt", ".pkl"))
        src_label = self._get_src_label(file, properties)

        id = "_".join(file.split("/")[-3:]).replace(".pt", "")
        data_dict = {
            "file_path": file,
            "image": data[:-1],
            "label": data[-1:],
            "src_label": src_label,
            "properties": properties,
            "id": id,
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def _get_src_label(self, file, properties):
        # source label is the label from the original dataset without any preprocessing
        src_label_path = file.replace(get_data_path(), get_source_labels_path()).replace(".pt", "_label.nii.gz")
        src_label_nii = read_file_to_nifti_or_np(src_label_path)
        src_label_nii = reorient_nib_image(
            src_label_nii,
            original_orientation=properties["original_orientation"],
            target_orientation=properties["new_direction"],
        )
        src_label_npy = nifti_or_np_to_np(src_label_nii)
        return torch.from_numpy(src_label_npy).float().unsqueeze(0).unsqueeze(0)


class ClsRegTestDataset(Dataset):
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
        data_dict = {
            "file_path": file,
            "image": data[0],
            "CLSREG_label": data[1],
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict
