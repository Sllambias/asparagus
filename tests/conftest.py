import lightning as L
import pickle
import pytest
import torch


@pytest.fixture
def pretrain_files(tmp_path):
    """Three .pt files of shape [1, 32, 32, 32] for pretraining (raw image, no label).
    32^3 ensures the UNet bottleneck (4 max-pool stages) stays at 2x2x2, avoiding
    single-element BatchNorm errors with batch_size=1.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"pre_{i:03d}.pt"
        torch.save(torch.randn(1, 32, 32, 32), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def seg_files(tmp_path):
    """Three .pt + .pkl file pairs for segmentation. Shape [2, 32, 32, 32] = [image, label].
    32^3 ensures the UNet bottleneck (4 max-pool stages) stays at 2x2x2.
    """
    files = []
    for i in range(3):
        pt = tmp_path / f"seg_{i:03d}.pt"
        pkl = tmp_path / f"seg_{i:03d}.pkl"
        data = torch.zeros(2, 32, 32, 32)
        data[0] = torch.randn(32, 32, 32)
        data[1] = torch.randint(0, 2, (32, 32, 32)).float()
        torch.save(data, pt)
        with open(pkl, "wb") as f:
            pickle.dump({"foreground_locations": []}, f)
        files.append(str(pt))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def clsreg_files(tmp_path):
    """Three .pt files containing (image[1,32,32,32], label_scalar) tuples.
    32^3 prevents single-element BatchNorm errors in the 4-stage UNet encoder.
    Labels are 0-dim int tensors; ClassificationModule.on_before_batch_transfer
    squeezes and converts to long before the training step.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"cls_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32, 32), torch.tensor(i % 2)), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]], "test": [files[2]]}


@pytest.fixture
def reg_files(tmp_path):
    """Three .pt files containing (image[1,32,32,32], label[1]) tuples.
    Labels are 1D float tensors so they collate to [B, 1], matching the
    unet_clsreg_tiny output shape [B, 1] expected by MeanSquaredError.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"reg_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32, 32), torch.tensor([float(i % 2)])), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]], "test": [files[2]]}


@pytest.fixture
def cls_probe_files(tmp_path):
    """Five .pt files for classification / linear-probe tests. 0-dim integer labels.
    2 train + 2 val gives full batches when batch_size=2, avoiding the squeeze()-to-scalar
    edge case in ClassificationModule.on_before_batch_transfer with batch_size=1.
    2 test files (labels 1, 0) ensure both classes are present for AUROC computation.
    """
    labels = [0, 1, 0, 1, 0, 1]
    files = []
    for i, lbl in enumerate(labels):
        path = tmp_path / f"clsp_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32, 32), torch.tensor(lbl)), path)
        files.append(str(path))
    return {"train": files[:2], "val": files[2:4], "test": files[4:6]}


@pytest.fixture
def pretrain_files_2d(tmp_path):
    """Three .pt files of shape [1, 32, 32] for 2D pretraining (raw image, no label).
    2D analogue of pretrain_files; 32^2 keeps the tiny UNet bottleneck at 8x8.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"pre2d_{i:03d}.pt"
        torch.save(torch.randn(1, 32, 32), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def seg_files_2d(tmp_path):
    """Three .pt + .pkl file pairs for 2D segmentation. Shape [2, 32, 32] = [image, label].
    2D analogue of seg_files; .pkl sidecar required by SegDataModule.
    """
    files = []
    for i in range(3):
        pt = tmp_path / f"seg2d_{i:03d}.pt"
        pkl = tmp_path / f"seg2d_{i:03d}.pkl"
        data = torch.zeros(2, 32, 32)
        data[0] = torch.randn(32, 32)
        data[1] = torch.randint(0, 2, (32, 32)).float()
        torch.save(data, pt)
        with open(pkl, "wb") as f:
            pickle.dump({"foreground_locations": []}, f)
        files.append(str(pt))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def cls_probe_files_2d(tmp_path):
    """Six .pt files for 2D classification / linear-probe tests. 0-dim integer labels.
    2D analogue of cls_probe_files; same label pattern [0,1,0,1,0,1].
    """
    labels = [0, 1, 0, 1, 0, 1]
    files = []
    for i, lbl in enumerate(labels):
        path = tmp_path / f"clsp2d_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32), torch.tensor(lbl)), path)
        files.append(str(path))
    return {"train": files[:2], "val": files[2:4], "test": files[4:6]}


@pytest.fixture
def reg_files_2d(tmp_path):
    """Three .pt files containing (image[1,32,32], label[1]) tuples.
    2D analogue of reg_files; labels are 1D float tensors so they collate to [B, 1].
    """
    files = []
    for i in range(3):
        path = tmp_path / f"reg2d_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32), torch.tensor([float(i % 2)])), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]], "test": [files[2]]}


@pytest.fixture
def make_trainer(tmp_path):
    """Factory fixture that builds a minimal CPU Trainer for smoke tests."""

    def _make(**kwargs):
        defaults = dict(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=5,
            limit_val_batches=5,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        defaults.update(kwargs)
        return L.Trainer(default_root_dir=str(tmp_path), **defaults)

    return _make
