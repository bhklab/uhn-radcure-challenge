import os
from typing import Callable, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed

import torch
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

    The images are loaded using SimpleITK, preprocessed and cached for faster
    retrieval during training.
    """
    def __init__(self,
                 root_directory: str,
                 clinical_data_path: str,
                 patch_size: int = 50,
                 target_col: str = "target_binary",
                 train: bool = True,
                 cache_dir: str = "data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1):
        """Initialize the class.

        If the cache directory does not exist, the dataset is first
        preprocessed and cached.

        Parameters
        ----------
        root_directory
            Path to directory containing the training and test images and
            segmentation masks.
        clinical_data_path
            Path to a CSV file with subject metadata and clinical information.
        patch_size
            The size of isotropic patch to extract around the tumour centre.
        target_col
            The name of a column in the clinical dataframe used as prediction
            target.
        train
            Whether to load the training or test set.
        cache_dir
            Path to directory where the preprocessed images will be cached.
        transform
            Callable used to transform the images after preprocessing.
        num_workers
            Number of parallel processes to use for data preprocessing.
        """
        self.root_directory = root_directory
        self.patch_size = patch_size
        self.target_col = target_col
        self.train = train
        self.transform = transform
        self.num_workers = num_workers
                
        if self.train:
            self.split = "training"
        else:
            self.split = "test"
            
        clinical_data = pd.read_csv(clinical_data_path)
        try:
            self.clinical_data = clinical_data[clinical_data["split"] == self.split]
        except:
            self.clinical_data = clinical_data
            
        self.cache_path = os.path.join(cache_dir, self.split)

        if not self.train and len(self.clinical_data) == 0:
            warn(("The test set is not available at this stage of the challenge."
                  " Testing will be disabled"), UserWarning)
        else:
            # TODO we should also re-create the cache when the patch size is changed
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._prepare_data()

    def _prepare_data(self):
        """Preprocess and cache the dataset."""
        Parallel(n_jobs=self.num_workers)(
            delayed(self._preprocess_subject)(subject_id)
            for subject_id in self.clinical_data["Study ID"])

    def _preprocess_subject(self, subject_id: str):
        """Preprocess and cache a single subject."""
        # load image and GTV mask
        path = os.path.join(self.root_directory, self.split,
                            "{}", f"{subject_id}.nrrd")
        image = sitk.ReadImage(path.format("images"))
        mask = sitk.ReadImage(path.format("masks"))

        # crop the image to (patch_size)^3 patch around the tumour centre
        tumour_centre = find_centroid(mask)
        size = np.ceil(self.patch_size / np.asarray(image.GetSpacing())).astype(np.int) + 1
        min_coords = np.floor(tumour_centre - size / 2).astype(np.int64)
        max_coords = np.floor(tumour_centre + size / 2).astype(np.int64)
        min_x, min_y, min_z = min_coords
        max_x, max_y, max_z = max_coords
        image = image[min_x:max_x, min_y:max_y, min_z:max_z]

        # resample to isotropic 1 mm spacing
        reference_image = sitk.Image([self.patch_size]*3, sitk.sitkFloat32)
        reference_image.SetOrigin(image.GetOrigin())
        image = sitk.Resample(image, reference_image)

        # window image intensities to [-500, 1000] HU range
        image = sitk.Clamp(image, sitk.sitkFloat32, -500, 1000)

        sitk.WriteImage(image, os.path.join(self.cache_path, f"{subject_id}.nrrd"), True)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        subject_id = self.clinical_data.iloc[idx]["Study ID"]
        target = self.clinical_data.iloc[idx][self.target_col]

        path = os.path.join(self.cache_path, f"{subject_id}.nrrd")
        image = sitk.ReadImage(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)



