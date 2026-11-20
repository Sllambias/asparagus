from asparagus.functional.loading import load_image_file
from asparagus.modules.datasets.TrainDataset import FileDataset


class PretrainDataset(FileDataset):
    def __getitem__(self, idx):
        file = self.files[idx]
        data = load_image_file(file)
        data_dict = {"file_path": file, "image": data, "transforms_applied": {}}
        data_dict = self._transform(data_dict)  # CPU transforms only here
        return data_dict
