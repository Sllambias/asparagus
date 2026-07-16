import nibabel as nib
import numpy as np
import torch
from PIL import Image


def load_image_file(file: str) -> torch.Tensor:
    if file.endswith(".pt"):
        return torch.load(file)
    elif file.endswith(".nii.gz") or file.endswith(".nii"):
        nii = nib.load(file)
        data = nii.get_fdata(dtype=np.float32)
        tensor = torch.from_numpy(data)
        return tensor.unsqueeze(0)  # (H,W,D) -> (1,H,W,D) to match .pt channel convention
    elif file.endswith(".png"):
        image = Image.open(file)
        if image.mode == "RGB":
            image = image.convert("L")
        image = np.array(image)
        image = torch.tensor(image)
        return image.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported file format: {file}. Expected .pt, .png, .nii, or .nii.gz and found {file}")
