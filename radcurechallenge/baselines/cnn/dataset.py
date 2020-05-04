import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

from torch.utils.data import Dataset


def find_centroid(mask: sitk.Image) -> np.ndarray:
    """Find the centroid of a binary image in image
    coordinates.

    Parameters
    ----------
    mask
        The bimary mask image.

    Returns
    -------
    np.ndarray
        The (x, y, z) coordinates of the centroid
        in image space.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
    return np.asarray(centroid_idx, dtype=np.float64)


class RadcureDataset(Dataset):
    """Dataset class used in simple CNN baseline training.

    The images are loaded using SimpleITK and """
    def __init__(self,
                 root_directory,
                 clinical_data_path,
                 patch_size=50,
                 target_col="target_binary",
                 train=True,
                 cache_dir=".cache/",
                 transform=None):
        self.root_directory = root_directory
        self.patch_size = patch_size
        self.train = train
        self.cache_dir = cache_dir
        self.transform = transform

        if self.train:
            self.split = "training"
        else:
            self.split = "validation"

        clinical_data = pd.read_csv(clinical_data_path)
        self.clinical_data = clinical_data[clinical_data["split"] == self.split]

        if self.cache_dir:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                self.cached = set()
            else:
                self.cached = set([os.path.splitext(f)[0] for f in os.listdir(self.cache_dir)])

    def _load_from_disk(self, subject_id):
        # load image and GTV mask
        path = os.path.join(self.root_directory, self.split, "{}", f"{subject_id}.nrrd")
        image, mask = sitk.ReadImage(path.format("images")), sitk.ReadImage(path.format("masks"))

        # crop the image to (patch_size)^3 patch around the tumour centre
        tumour_centre = find_centroid(mask)
        size = np.ceil(self.patch_size / np.asarray(image.GetSpacing())).astype(np.int)
        min_coords = np.floor(tumour_centre - size / 2).astype(np.int64)
        max_coords = np.floor(tumour_centre + size / 2).astype(np.int64)
        min_x, min_y, min_z = min_coords
        max_x, max_y, max_z = max_coords
        image = image[min_x:max_x, min_y:max_y, min_z:max_z]

        # resample to isotropic 1 mm spacing
        reference_image = sitk.Image([self.patch_size]*3, sitk.sitkFloat32)
        reference_image.SetOrigin(image.GetOrigin())
        image = sitk.Resample(image, reference_image)

        image = sitk.Clamp(image, -500, 1000)

        return image

    def _load_from_cache(self, subject_id):
        path = os.path.join(self.cache_dir, f"{subject_id}.nrrd")
        return sitk.ReadImage(path)

    def __getitem__(self, idx):
        subject_id = self.clinical_data.iloc[idx]["Study ID"]
        target = self.clinical_data.iloc[idx][self.target_col]

        if self.cache_dir and subject_id in self.cached:
            image = self._load_from_cache(subject_id)
            sitk.WriteImage(image, os.path.join(self.cache_dir, f"{subject_id}.nrrd"))
            self.cached.add(subject_id)
        else:
            image = self._load_from_disk(subject_id)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.clinical_data)



