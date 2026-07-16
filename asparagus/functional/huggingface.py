"""Download and load models from HuggingFace Hub."""

from huggingface_hub import hf_hub_download, list_repo_files


def download_hf_checkpoint(repo_id: str) -> str:
    """Download a checkpoint file from HuggingFace Hub. Returns local path."""
    all_files = list_repo_files(repo_id)
    checkpoint_files = [f for f in all_files if f.endswith((".ckpt", ".pt", ".pth"))]

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {repo_id}. Available: {all_files}")

    filename = checkpoint_files[0]
    print(f"Downloading {filename} from {repo_id}...")

    return hf_hub_download(repo_id, filename)


class HuggingFaceWeightMapper:
    """Remaps weights to asparagus format."""

    def __init__(self, state_dict: dict):
        self.state_dict = state_dict

    def remap_keys(self) -> dict:
        """Add 'model.' prefix for Lightning module compatibility. Subclasses should override and call super()."""
        first_key = next(iter(self.state_dict.keys()))
        if not first_key.startswith("model."):
            return {f"model.{k}": v for k, v in self.state_dict.items()}
        return self.state_dict


class OpenMindResEncWeightMapper(HuggingFaceWeightMapper):
    """Remaps OpenMind ResEncUNet weights to asparagus format."""

    def remap_keys(self) -> dict:
        original_keys = self.state_dict.keys()
        self.state_dict = {
            k.replace(".convs.0.", ".conv1.").replace(".norm.", ".norm_op."): v for k, v in self.state_dict.items()
        }
        if self.state_dict.keys() != original_keys:
            print("Remapped OpenMind ResEncUNet keys to asparagus naming conventions.")

        return super().remap_keys()


class OpenMindPrimusWeightMapper(HuggingFaceWeightMapper):
    """Remaps OpenMind Primus weights to asparagus format."""

    def remap_keys(self) -> dict:
        original_keys = self.state_dict.keys()
        self.state_dict = {
            k.replace("encoder.eva.", "eva.")
            .replace("encoder.down_projection.proj.", "encoder.proj.")
            .replace("encoder.mask_token", "mask_token"): v
            for k, v in self.state_dict.items()
        }
        if self.state_dict.keys() != original_keys:
            print("Remapped OpenMind Primus keys to asparagus naming conventions.")

        return super().remap_keys()
