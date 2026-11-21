import torch
from lightning.fabric.utilities.seed import pl_worker_init_function
from monai.transforms import Randomizable


def seed_worker(worker_id: int) -> None:
    """Seed a data loader worker like Lightning does, plus MONAI's shared random state, which forked workers all copy."""
    pl_worker_init_function(worker_id)
    Randomizable.R.seed(torch.initial_seed() % 2**32)
