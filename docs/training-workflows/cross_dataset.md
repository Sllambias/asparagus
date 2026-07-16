# Cross-Dataset Evaluation

This page covers workflows where the **training** and **test** datasets differ — for example, training a model on dataset A and evaluating its generalisation on dataset B.

## Train on A, Test on B

Asparagus supports specifying a separate test task at training time via the `test_task` parameter. When provided, inference is automatically run on `test_task` using the best checkpoint once training finishes.

```bash
asp_train_seg \  # or asp_train_reg / asp_train_cls
    task=TASK_A \
    +model=unet_b_lw_dec \
    data.train_split=split_40_10_50 \
    test_task=TASK_B \
    data.test_split=paths
```

### Parameters

| Parameter | Description |
|---|---|
| `task` | Task used for training and validation (dataset A) |
| `+model` | Model architecture config |
| `data.train_split` | Split file for training/validation (e.g. `split_40_10_50`) |
| `test_task` | Task to run inference on after training (dataset B) |
| `data.test_split` | Split file for the test task (see below) |

!!! note "Preprocessing both datasets"
    Both `TASK_A` and `TASK_B` must be preprocessed before running this command:

    1. Preprocess dataset A (`TASK_A`) — used for training and validation.
    2. Preprocess dataset B (`TASK_B`) — used for testing only.

    See [Preprocessing](../data-pipeline/preprocessing.md) for details.


## Test on Entire Dataset

!!! warning "Pending"
    Full-dataset test support via the `paths` split is currently pending.

When you want to evaluate on **all samples** in a dataset (rather than a held-out test split), use `paths` as the split file name:

```bash
data.test_split=paths
```

The `paths` file lists every sample in the dataset without any train/val/test partitioning, so setting `data.test_split=paths` effectively runs inference on the entire dataset B.

This is particularly useful for:

- Generating predictions for an external dataset where no predefined split exists.
- Evaluating generalisation on a fully independent cohort.
