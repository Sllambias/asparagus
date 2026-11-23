import logging
import nibabel as nib
import numpy as np
import os
import polars
import torch
import torchvision
from asparagus.functional.loading import load_image_file
from asparagus.paths import get_data_path, get_source_labels_path
from gardening_tools.functional.nibabel_utils import reorient_nib_image
from gardening_tools.functional.paths.read import load_pickle, read_file_to_nifti_or_np
from gardening_tools.functional.type_conversions import nifti_or_np_to_np
from torch.utils.data import Dataset
from typing import Any, Dict, Optional


class FileDataset(Dataset):
    """Samples loaded from a list of files, each passed through the optional transforms."""

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

    def get_existing_file(self, idx: int) -> str:
        """Return the file at idx, or the next file that exists if it is missing."""
        for offset in range(len(self.files)):
            file = self.files[(idx + offset) % len(self.files)]
            if os.path.isfile(file):
                return file
            logging.debug(f"Skipping missing file: {file}")
        raise FileNotFoundError("None of the files in the dataset exist")

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict


class SegDataset(FileDataset):
    def __getitem__(self, idx):
        file = self.get_existing_file(idx)
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
        data_dict = super()._transform(data_dict)
        data_dict.pop("foreground_locations")
        return data_dict


class ClsRegDataset(FileDataset):
    def __getitem__(self, idx):
        file = self.get_existing_file(idx)
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data[0],
            "CLSREG_label": data[1],
            "transforms_applied": {},
        }

        return self._transform(data_dict)


class TableClsRegDataset(FileDataset):
    """Image files whose label is in a table, found by file path. Labels below label_cutoff become 1 and the rest 0."""

    def __init__(
        self,
        files: list,
        table: polars.DataFrame,
        label_cutoff: float,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        """
        Args:
            files: Image files.
            table: The columns file_path and label, sorted by file_path.
            label_cutoff: Labels below it become 1 and the rest 0.
            transforms: Transforms of the sample dict.
        """
        super().__init__(files, transforms=transforms)
        self.table = table
        self.label_cutoff = label_cutoff

    def __getitem__(self, idx):
        return self.load_sample(self.get_existing_file(idx))

    def load_sample(self, file: str) -> Dict[str, Any]:
        row = self.table["file_path"].search_sorted(file)
        assert row < len(self.table) and self.table["file_path"][row] == file, f"{file} is not in the label table"
        data_dict = {
            "file_path": file,
            "image": load_image_file(file),
            "CLSREG_label": torch.tensor([float(self.table["label"][row] < self.label_cutoff)]),
            "transforms_applied": {},
        }
        return self._transform(data_dict)


class TableClsRegTestDataset(TableClsRegDataset):
    def __getitem__(self, idx):
        return self.load_sample(self.files[idx])


class SegTestDataset(FileDataset):
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


class ClsRegTestDataset(FileDataset):
    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data[0],
            "CLSREG_label": data[1],
        }

        return self._transform(data_dict)


class SingleSubjectPredictDataset(FileDataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        properties = {}
        all_channels = []
        for file in self.files:
            if file.endswith(".pt"):
                data = torch.load(file)
            elif file.endswith(".npy"):
                data = torch.from_numpy(np.load(file))
            elif file.endswith(".nii") or file.endswith(".nii.gz"):
                data = nib.load(file)
                properties["nifti_metadata"] = {
                    "affine": data.affine,
                    "header": data.header,
                    "reoriented": False,
                }
                data = torch.from_numpy(data.get_fdata()[np.newaxis])
            else:
                raise ValueError(f"Unsupported file type: {file}")
            data = data.float()
            all_channels.append(data)

        data = torch.vstack(all_channels)
        properties["original_size"] = data.shape[1:]  # Exclude channel dimension
        data_dict = {
            "file_path": file,
            "image": data,
            "properties": properties,
        }

        return self._transform(data_dict)
