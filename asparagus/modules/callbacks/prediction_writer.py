import os
from gardening_tools.functional.paths.write import save_prediction_from_logits
from lightning.pytorch.callbacks import BasePredictionWriter


class WritePredictionFromLogits(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: str = "batch", save_format: str = ".nii.gz"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_format = save_format

    def write_on_batch_end(self, _trainer, _pl_module, data_dict, _batch_indices, _batch, _batch_idx, _dataloader_idx):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        logits, properties, case_id = (
            data_dict["logits"],
            data_dict["properties"],
            data_dict["id"],
        )

        save_prediction_from_logits(
            logits,
            os.path.join(self.output_dir, case_id),
            properties=properties,
            save_format=self.save_format,
        )
        del data_dict
