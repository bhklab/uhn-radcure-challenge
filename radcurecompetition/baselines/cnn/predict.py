from .model import SimpleCNN
from .dataset import RadcureDataset


def predict(root_directory, weights_path):
    model = ""
    dataset = RadcureDataset(root_directory=root_directory,
                             clinical_data_path=clinical_data_path,
                             patch_size=)
