import os
import torch
from asparagus.functional.huggingface import download_hf_checkpoint
from asparagus.functional.versioning import detect_id
from hydra.utils import get_class


def load_checkpoint_state_dict(path):
    """Load a checkpoint file and return the state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        print(f"Loading weights trained for {ckpt.get('global_step', '?')} steps / {ckpt.get('epoch', '?')} epochs.")
        return ckpt["state_dict"]
    elif "network_weights" in ckpt:
        print("Loading weights from external checkpoint (network_weights key).")
        return ckpt["network_weights"]
    else:
        raise ValueError("Unsupported checkpoint format. Expected 'state_dict' or 'network_weights' key.")


def resolve_checkpoint_path(cfg):
    """Resolve checkpoint file path from config. Returns path or None."""
    if cfg.checkpoint_run_id:
        folder = detect_id(cfg.checkpoint_run_id)
        return os.path.join(folder, "checkpoints", cfg.load_checkpoint_name)
    if cfg.checkpoint_path:
        return cfg.checkpoint_path
    return None


def resolve_checkpoint(cfg):
    """Resolve and load checkpoint from config. Returns a state_dict or None."""
    hf_id = getattr(cfg, "hf_model_id", None) or None
    ckpt_path = resolve_checkpoint_path(cfg)

    sources = [s for s in [ckpt_path, hf_id] if s]
    if len(sources) > 1:
        raise ValueError("Provide only one of: checkpoint_run_id, checkpoint_path, hf_model_id")
    if len(sources) == 0:
        return None

    if ckpt_path:
        return load_checkpoint_state_dict(ckpt_path)

    path = download_hf_checkpoint(hf_id)
    state_dict = load_checkpoint_state_dict(path)

    weight_mapper = get_class(cfg.hf_weight_format)
    return weight_mapper(state_dict).remap_keys()
