"""Finetuning a DINOv2 (UltraDino) checkpoint for classification on PNGs whose labels are in a CSV table."""

import csv
import json
import numpy as np
import pytest
import torch
from asparagus.functional.lr_scheduling import sawtooth_warmup_cosine_decay_schedule, vit_layer_decay_param_groups
from asparagus.functional.sampling import class_sampling_weights
from asparagus.modules.data_modules.training import TableClsRegDataModule
from asparagus.modules.lightning_modules import ClassificationModule
from asparagus.modules.lightning_modules.dinov2 import DINOv2Module
from asparagus.modules.networks.dinov2 import DINOv2, DINOv2CLSREG
from asparagus.modules.transforms.presets import ultradino_clsreg_train, ultradino_clsreg_val_test
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader


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
        # In reverse order, so the labels have to be matched to the files by path and not by row
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
    with pytest.raises(AssertionError, match="have no label in the table"):
        data_module.setup("fit")


def test_forked_workers_load_batches(labelled_pngs):
    # polars hangs when it is called in a forked worker, so the datasets must not use it in __getitem__
    data_module = make_data_module(labelled_pngs)
    data_module.setup("fit")
    loader = DataLoader(data_module.train_dataset, batch_size=2, num_workers=2, multiprocessing_context="fork", timeout=30)
    labels = [label for batch in loader for label in batch["CLSREG_label"].flatten().tolist()]
    assert labels == [float(ga < 259) for ga in labelled_pngs["ga"][:2]]


def test_class_sampling_weights():
    labels = np.array([0, 0, 0, 1], dtype=np.float32)
    assert class_sampling_weights(labels).tolist() == [1, 1, 1, 1]
    assert class_sampling_weights(labels, class_factors=[1, 3]).tolist() == [1, 1, 1, 3]
    assert class_sampling_weights(labels, class_balance=1).tolist() == [1 / 3, 1 / 3, 1 / 3, 1]
    weights = class_sampling_weights(labels, class_factors=[1, 2], class_balance=0.5)
    assert weights.tolist() == pytest.approx([3**-0.5] * 3 + [2])
    with pytest.raises(AssertionError, match="class_factors has 1 values"):
        class_sampling_weights(labels, class_factors=[1])


@pytest.mark.parametrize(
    "class_factors, class_balance, preterm_share",
    [(None, 0, 2 / 6), (None, 1, 1 / 2), (OmegaConf.create([1, 4]), 0, 8 / 12), ([1, 2], 1, 2 / 3)],
)
def test_oversampling_draws_the_classes_with_the_configured_shares(labelled_pngs, class_factors, class_balance, preterm_share):
    files = labelled_pngs["train"] + labelled_pngs["val"] + labelled_pngs["test"]
    data_module = TableClsRegDataModule(
        label_table=labelled_pngs["label_table"],
        label_column="GA",
        # Two of the six images are preterm
        label_cutoff=235,
        class_factors=class_factors,
        class_balance=class_balance,
        batch_size=2,
        num_workers=2,
        train_split=files,
        val_split=files,
    )
    data_module.setup("fit")
    torch.manual_seed(0)
    indices = list(data_module.train_dataloader().sampler)
    labels = data_module.train_dataset.labels
    assert labels.tolist() == [1, 0, 0, 0, 1, 0]
    assert labels[indices].mean() == pytest.approx(preterm_share, abs=0.005)


def test_oversampling_needs_the_random_datasampler(labelled_pngs):
    with pytest.raises(AssertionError, match="need use_random_datasampler"):
        TableClsRegDataModule(
            label_table=labelled_pngs["label_table"],
            label_column="GA",
            label_cutoff=259,
            class_balance=1,
            batch_size=2,
            num_workers=2,
            train_split=labelled_pngs["train"],
            val_split=labelled_pngs["val"],
            use_random_datasampler=False,
        )


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


def test_layer_decay_groups_the_vit_by_depth(tiny_vit):
    model = DINOv2CLSREG(output_channels=2, num_register_tokens=2, layer_scale_init=1e-5, **tiny_vit)
    module = ClassificationModule(model=model, compile_mode="default")
    groups = vit_layer_decay_param_groups(module.named_parameters(), learning_rate=1.0, layer_decay=0.5)

    names = ["encoder_layer_0", "encoder_layer_1", "encoder_layer_2", "encoder_layer_3", "decoder"]
    assert [group["name"] for group in groups] == names
    assert [group["lr"] for group in groups] == [0.125, 0.25, 0.5, 1.0, 1.0]
    group_names = {}
    for group in groups:
        for param in group["params"]:
            group_names[id(param)] = group["name"]
    backbone = model.teacher_backbone
    assert group_names[id(backbone.vit.patch_embedding.patch_embeddings.weight)] == "encoder_layer_0"
    assert group_names[id(backbone.vit.patch_embedding.position_embeddings)] == "encoder_layer_0"
    assert group_names[id(backbone.vit.cls_token)] == "encoder_layer_0"
    assert group_names[id(backbone.register_tokens)] == "encoder_layer_0"
    assert group_names[id(backbone.vit.blocks[0].attn.qkv.weight)] == "encoder_layer_1"
    assert group_names[id(backbone.vit.blocks[1].ls2.gamma)] == "encoder_layer_2"
    assert group_names[id(backbone.vit.norm.weight)] == "encoder_layer_3"
    assert group_names[id(model.decoder[1].weight)] == "decoder"
    # Every trainable parameter is in one group, and the frozen mask token and MONAI head are in none
    trainable = [param for param in model.parameters() if param.requires_grad]
    assert sum(len(group["params"]) for group in groups) == len(trainable)
    assert set(group_names) == {id(param) for param in trainable}


def test_encoder_layers_stay_frozen_while_the_decoder_warms_up(tiny_vit):
    module = ClassificationModule(model=DINOv2CLSREG(output_channels=2, **tiny_vit))
    groups = vit_layer_decay_param_groups(module.named_parameters(), learning_rate=1.0, layer_decay=0.5)
    optimizer = torch.optim.AdamW(groups)
    scheduler = sawtooth_warmup_cosine_decay_schedule(
        optimizer,
        decoder_warmup_epochs=1,
        warmup_epochs=1,
        steps_per_epoch=10,
        cosine_period_ratio=1,
        max_epochs=4,
    )
    lrs = []
    for _ in range(40):
        lrs.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()

    base_lrs = [0.125, 0.25, 0.5, 1.0, 1.0]
    for step in range(10):
        assert lrs[step][:4] == [0.0] * 4
        assert lrs[step][4] == pytest.approx(0.999 * step / 10 + 0.001)
    # After the head-only phase the layers keep their ratios through the warmup and the cosine decay
    for step in [10, 15, 20, 30]:
        assert lrs[step] == pytest.approx([lrs[step][4] * base_lr for base_lr in base_lrs])
    assert lrs[20] == pytest.approx(base_lrs)


@pytest.mark.parametrize(
    "layer_decay, group_lrs",
    [
        # The default keeps one encoder and one decoder group
        (None, {"encoder": 1e-3, "decoder": 1e-3}),
        (
            0.5,
            {
                "encoder_layer_0": 1e-3 * 0.5**3,
                "encoder_layer_1": 1e-3 * 0.5**2,
                "encoder_layer_2": 1e-3 * 0.5,
                "encoder_layer_3": 1e-3,
                "decoder": 1e-3,
            },
        ),
    ],
)
def test_finetune_with_layer_decay(labelled_pngs, tiny_vit, make_trainer, tmp_path, layer_decay, group_lrs):
    module = ClassificationModule(
        model=DINOv2CLSREG(output_channels=2, **tiny_vit),
        layer_decay=layer_decay,
        learning_rate=1e-3,
        optimizer="AdamW",
        decoder_warmup_epochs=1,
        warmup_epochs=1,
        test_output_path=str(tmp_path / "predictions.json"),
    )

    trainer = make_trainer(max_epochs=3)
    trainer.fit(module, datamodule=make_data_module(labelled_pngs))

    param_groups = trainer.optimizers[0].param_groups
    assert {group["name"]: group["initial_lr"] for group in param_groups} == pytest.approx(group_lrs)
