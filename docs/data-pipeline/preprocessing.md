# Preprocessing

Asparagus trains on preprocessed data. Raw datasets must be converted into the expected format using the **asparagus_preprocessing** companion repository before you can start training.

This page walks you through the full preprocessing workflow step by step.


## Overview

```
Raw dataset  →  Write a preprocessing script  →  Run it  →  Create splits  →  Train
```

1. Write a dataset-specific preprocessing script
2. Run it to produce preprocessed files + metadata
3. Generate train/val/test splits
4. Point Asparagus at the task and start training


## Step 1 — Write a preprocessing script

Each dataset gets its own Python script placed in the appropriate subfolder of `asparagus_preprocessing/`:

| Task type | Subfolder |
|---|---|
| Pretraining | `datasets_pretraining/` |
| Segmentation | `datasets_segmentation/` |
| Classification | `datasets_classification/` |
| Regression | `datasets_regression/` |

Name the file after your task (e.g. `SEG005_MyDataset.py`). A blank template is available at `datasets_pretraining/TEMPLATE.py`.

### Naming convention

Task names must follow `<PREFIX><XXX>_<Name>`:

| Task type | Prefix | Example |
|---|---|---|
| Pretraining | `PT` | `PT003_BrainMRI` |
| Segmentation | `SEG` | `SEG005_MyDataset` |
| Classification | `CLS` | `CLS003_SALD` |
| Regression | `REGR` | `REGR005_Age` |

The three-digit number must be unique within each task type (e.g. `PT005` and `SEG005` can coexist).

### Script structure

Every preprocessing script follows the same five-step pattern:

```python
# Step 0 — define main() with default arguments
def main(path=get_source_path(), subdir="MyDataset", processes=12, ...):

    # Step 1 — define configs
    dataset_config      = DatasetConfig(task_name="SEG005_MyDataset", ...)
    saving_config       = get_FOMO300K_saving_config(save_as_tensor=True, ...)
    preprocessing_config = get_noresampling_preprocessing_config()

    # Step 2 — set up source / target paths
    source_dir = os.path.join(path, subdir)
    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    os.makedirs(target_dir, exist_ok=True)

    # Step 3 — find input files
    files_standard, files_DWI, files_PET, files_Perf, files_excluded = \
        recursive_find_and_group_files(source_dir, ...)
    files_standard_out = get_image_output_paths(files_standard, source_dir, target_dir, ...)

    # Step 4 — process the dataset
    process_dataset_without_table(process_fn=process_sample, ...)  # segmentation
    # or
    process_dataset_with_table(process_fn=process_sample, ...)     # cls / regression

    # Step 5 — postprocess (generates dataset.json and paths.json)
    postprocess_standard_dataset(dataset_config=dataset_config, ...)
```

See the [asparagus_preprocessing repo](https://github.com/Sllambias/asparagus_preprocessing) for example scripts covering each task type.

### Key configs at a glance

**DatasetConfig** — describes the dataset:

| Field | Description |
|---|---|
| `task_name` | Unique task name (e.g. `SEG005_MyDataset`) |
| `n_modalities` | Number of input channels (1 for single-modality MRI) |
| `n_classes` | Number of output classes (or `1` for regression) |
| `in_extensions` | File extensions to look for (e.g. `[".nii.gz"]`) |
| `patterns_exclusion` | Filename patterns to skip (e.g. labels, unwanted sequences) |

**PreprocessingConfig presets** — pick one and customise:

| Preset function | Spacing | Use case |
|---|---|---|
| `get_noresampling_preprocessing_config()` | Native | Keep original voxel spacing |
| `get_iso_preprocessing_config()` | 1 mm isotropic | Resample to 1×1×1 mm |

**SavingConfig** — `get_FOMO300K_saving_config(save_as_tensor=True, ...)` is the standard choice for most datasets.

??? info "Full config dataclass definitions"
    **DatasetConfig**
    ```python
    @dataclass
    class DatasetConfig:
        df_columns: list
        task_name: str
        n_classes: int
        n_modalities: int
        in_extensions: str
        patterns_exclusion: list
        patterns_DWI: list
        patterns_PET: list
        patterns_perfusion: list
        patterns_m0: list
        patterns_bidsify: list
        split: str
    ```

    **SavingConfig**
    ```python
    @dataclass
    class SavingConfig:
        save_as_tensor: bool
        tensor_dtype: str
        bidsify: bool
        save_dset_metadata: bool
        save_file_metadata: bool  # must be True for segmentation
    ```

    **PreprocessingConfig**
    ```python
    @dataclass
    class PreprocessingConfig:
        normalization_operation: List   # one entry per modality
        target_spacing: Optional[List]
        background_pixel_value: int = 0
        crop_to_nonzero: bool = True
        keep_aspect_ratio_when_using_target_size: bool = False
        image_properties: Optional[dict] = field(default_factory=dict)
        intensities: Optional[List] = None
        target_orientation: Optional[str] = "RAS"
        target_size: Optional[List] = None
        min_slices: int = 0
        remove_nans: bool = True
    ```


### Saving raw labels (segmentation)

For segmentation tasks, the `process_sample` function must also save the original label to `$ASPARAGUS_RAW_LABELS` so that final test metrics can be computed against native labels:

```python
from asparagus_preprocessing.utils.saving import save_raw_label, save_modified_label

# If the label map is used as-is:
save_raw_label(file_out, label_path)

# If label values were remapped (e.g. collapsing classes), save the remapped-but-unprocessed version:
save_modified_label(file_out, label_arr)
```

When in doubt, use `save_modified_label`. Resampling or any spatial processing should **not** be applied to the raw label.

### Segmentation metadata (.pkl)

For each segmentation sample, a `.pkl` file must be saved alongside the `.pt` file (same path, different extension). It must contain at minimum:

```python
{"foreground_locations": [...]}  # indices of non-zero labels; may be empty
```

Set `save_file_metadata=True` in `SavingConfig` to have this handled automatically. The indices are used to oversample underrepresented classes during training.


## Step 2 — Run the script

Use the `asp_preprocess` CLI entry point, which automatically finds the right module by task name:

```bash
asp_preprocess \
    --dataset SEG005_MyDataset \
    --save_as_tensor \
    --num_workers 12
```

This produces under `$ASPARAGUS_DATA/SEG005_MyDataset/`:

```
SEG005_MyDataset/
├── dataset.json     ← task metadata (n_classes, n_modalities, preprocessing config, …)
├── paths.json       ← paths to all processed samples (used for splitting)
└── <subject dirs>/
    ├── file.pt      ← preprocessed image tensor
    └── file.pkl     ← per-file metadata (required for segmentation)
```


## Step 3 — Create train/val/test splits

Once `paths.json` exists, generate a split file with `asp_split`:

```bash
# Simple percentage split: 75% train, 15% val, 10% test
asp_split --dataset SEG005_MyDataset --vals 75 15 10
```

This saves `split_75_15_10.json` inside the task directory. The three numbers must sum to 100.

To use a predefined splitting strategy (e.g. subject-level stratification):

```bash
asp_split --dataset SEG005_MyDataset --fn BIDSsplit_40_10_50
```

!!! note "Split on subject level"
    The default `--vals` split operates on file level. If subjects have multiple scans, use `--fn` with a stratified function that groups by subject ID first.


## Step 4 — Train

With preprocessing done and a split file created, you are ready to train:

```bash
asp_train_seg \
    task=SEG005_MyDataset \
    +model=unet_b \
    data.train_split=split_75_15_10
```

See the task-specific training pages for full details:

- [Segmentation →](../training-workflows/segmentation.md)
- [Classification →](../training-workflows/classification.md)
- [Regression →](../training-workflows/regression.md)
