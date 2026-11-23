"""Finetuning a DINOv2 (UltraDino) checkpoint for classification on PNGs whose labels are in a CSV table."""

import csv
import json
import pytest
import torch
from asparagus.modules.data_modules.training import TableClsRegDataModule
from asparagus.modules.lightning_modules import ClassificationModule
from asparagus.modules.lightning_modules.dinov2 import DINOv2Module
from asparagus.modules.networks.dinov2 import DINOv2, DINOv2CLSREG
from asparagus.modules.transforms.presets import ultradino_clsreg_train, ultradino_clsreg_val_test
from PIL import Image


@pytest.fixture
def tiny_vit():
    return dict(img_size=(32, 32), patch_size=(8, 8), hidden_size=48, num_layers=2, num_heads=4, mlp_dim=96)


@pytest.fixture
def labelled_pngs(tmp_path):
    """Six 8-bit PNGs of different sizes and a label table with their GA at birth in days."""
    ga = [230, 270, 240, 280, 200, 265]
    generator = torch.Generator().manual_seed(0)
    files = []
    for i, shape in enumerate([(40, 50), (60, 30), (32, 32), (45, 70), (50, 40), (36, 64)]):
        path = str(tmp_path / f"us_{i}.png")
        Image.fromarray(torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).numpy()).save(path)
        files.append(path)
    label_table = tmp_path / "labels.csv"
    with open(label_table, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CPR_MOR", "file_path", "GA"])
        # In reverse order, so the data module has to sort the table
        for i, (file, file_ga) in reversed(list(enumerate(zip(files, ga)))):
            writer.writerow([f"mother_{i}", file, file_ga])
    return {"train": files[:2], "val": files[2:4], "test": files[4:], "label_table": str(label_table), "ga": ga}


def make_data_module(files):
    return TableClsRegDataModule(
        label_table=files["label_table"],
        label_column="GA",
        label_cutoff=259,
        batch_size=2,
        num_workers=2,
        train_split=files["train"],
        val_split=files["val"],
        test_samples=files["test"],
        train_transforms=ultradino_clsreg_train(target_size=(32, 32)),
        val_transforms=ultradino_clsreg_val_test(target_size=(32, 32)),
        test_transforms=ultradino_clsreg_val_test(target_size=(32, 32)),
        use_random_datasampler=False,
    )


def test_labels_are_binarized_at_the_cutoff(labelled_pngs):
    data_module = make_data_module(labelled_pngs)
    data_module.setup("fit")
    data_module.setup("test")
    datasets = [data_module.train_dataset, data_module.val_dataset, data_module.test_dataset]
    samples = [dataset[i] for dataset in datasets for i in range(len(dataset))]

    assert [sample["CLSREG_label"].item() for sample in samples] == [float(ga < 259) for ga in labelled_pngs["ga"]]
    for sample in samples:
        assert sample["image"].shape == (1, 32, 32) and sample["image"].dtype == torch.float32
    # Normalized with the FUS13M mean and std from [0, 1]
    assert samples[0]["image"].min() >= -0.1842924807 / 0.2187705424 - 1e-5


def test_files_missing_from_the_label_table_fail(labelled_pngs, tmp_path):
    unlabelled = str(tmp_path / "unlabelled.png")
    Image.fromarray(torch.zeros(32, 32, dtype=torch.uint8).numpy()).save(unlabelled)
    data_module = make_data_module(dict(labelled_pngs, train=[unlabelled]))
    data_module.setup("fit")
    with pytest.raises(AssertionError, match="is not in the label table"):
        data_module.train_dataset[0]


def test_dinov2_clsreg_loads_the_pretrained_teacher(tiny_vit):
    torch.manual_seed(0)
    pretrained = DINOv2(num_register_tokens=2, layer_scale_init=1e-5, projection_dim=64, **tiny_vit)
    with torch.no_grad():
        for param in pretrained.teacher_backbone.parameters():
            param.add_(1.0)
    weights = DINOv2Module(model=pretrained).state_dict()

    model = DINOv2CLSREG(output_channels=2, num_register_tokens=2, layer_scale_init=1e-5, **tiny_vit)
    module = ClassificationModule(model=model, weights=weights)

    for name, param in module.state_dict().items():
        if name.startswith("model.teacher_backbone."):
            assert torch.equal(param, weights[name]), name
    assert not model.teacher_backbone.mask_token.requires_grad


def test_finetune_and_test(labelled_pngs, tiny_vit, make_trainer, tmp_path):
    output_path = tmp_path / "predictions.json"
    model = DINOv2CLSREG(output_channels=2, dropout_rate=0.2, **tiny_vit)
    module = ClassificationModule(model=model, learning_rate=1e-3, warmup_epochs=0, test_output_path=str(output_path))
    data_module = make_data_module(labelled_pngs)

    trainer = make_trainer()
    trainer.fit(module, datamodule=data_module)
    trainer.test(module, datamodule=data_module)

    predictions = json.loads(output_path.read_text())
    labels = {file: predictions[file]["label"] for file in labelled_pngs["test"]}
    assert labels == {file: int(ga < 259) for file, ga in zip(labelled_pngs["test"], labelled_pngs["ga"][4:])}
