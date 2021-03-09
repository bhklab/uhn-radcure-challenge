import os
from typing import Callable, Optional
from collections import namedtuple

import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset

from .utils import encode_survival


Data = namedtuple("Data", ["image",
                           "emr",
                           "survival",
                           "survival_time",
                           "event",
                           "target_binary",
                           "mask"])


def load_emr_data(path: str, split: str = "training"):
    """Load and preprocess the EMR data."""
    emr_data = (pd.read_csv(path, index_col="Study ID")
                .query("split == @split")
                .drop("split", axis=1))
    if split == "training":
        emr_data["target_binary"] = emr_data["target_binary"].astype(np.uint8)
        emr_data["death"] = emr_data["death"].astype(np.uint8)
        emr_data["target_binary"] = (emr_data["target_binary"] & emr_data["death"]).astype(np.uint8)
        emr_data["survival_time"] *= 12

    # binarize T stage as T1/2 = 0, T3/4 = 1
    emr_data["T Stage"] = emr_data["T Stage"].map({
        "T1":     "T1/2",
        "T1a":    "T1/2",
        "T1b":    "T1/2",
        "T2":     "T1/2",
        "T3":     "T3/4",
        "T4":     "T3/4",
        "T4a":    "T3/4",
        "T4b":    "T3/4"
    })
    # use more fine-grained grouping for N stage
    emr_data["N Stage"] = emr_data["N Stage"].map({
        "N0":  "N0",
        "N1":  "N1",
        "N2":  "N2",
        "N2a": "N2",
        "N2b": "N2",
        "N2c": "N2",
        "N3":  "N3",
        "N3a": "N3",
        "N3b": "N3"
    })
    emr_data["Stage"] = emr_data["Stage"].map({
        "I": "I/II",
        "II": "I/II",
        "IIA": "I/II",
        "III": "III/IV",
        "IIIA": "III/IV",
        "IIIB": "III/IV",
        "IIIC": "III/IV",
        "IV": "III/IV",
        "IVA": "III/IV",
        "IVB": "III/IV"
    })
    emr_data["ECOG"] = emr_data["ECOG"] > 0
    emr_data = pd.get_dummies(emr_data,
                              columns=["Sex",
                                       "T Stage",
                                       "N Stage",
                                       "Disease Site",
                                       "Stage",
                                       "ECOG"],
                              drop_first=True)
    emr_data = pd.get_dummies(emr_data, columns=["HPV Combined"])
    return emr_data


class RadcureDataset(Dataset):
    """Dataset class used in simple CNN baseline training.
    """
    def __init__(self,
                 root_directory: str,
                 emr_data_path: str,
                 time_bins: np.ndarray,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 emr_transform: Optional[Callable] = None):
        """Initialize the object.

        Parameters
        ----------
        root_directory
            Path to directory containing the training and test images and
            segmentation masks.
        emr_data_path
            Path to a CSV file with EMR and outcome data.
        time_bins
            The time bins used in MTLR training.
        train
            Whether to load the training or test set.
        transform
            Callable used to transform the images after preprocessing.
        emr_transform
            Callable used to transform the EMR features.
        """
        self.root_directory = root_directory
        self.time_bins = time_bins
        self.num_time_bins = len(self.time_bins)
        self.transform = transform
        self.emr_transform = emr_transform

        self.split = "training" if train else "test"
        self.emr_data = load_emr_data(emr_data_path, self.split)
        if self.split == "training":
            self.targets = self.emr_data[["survival_time", "death", "target_binary"]]
        else:
            self.targets = None
        self.emr_data.drop(["survival_time", "death", "cancer_death",
                            "target_binary", "split"],
                           axis=1, errors="ignore", inplace=True)

    def __getitem__(self, idx):
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple
            The inputs and targets.
        """
        subject_id = self.emr_data.iloc[idx].name
        if self.split == "training":
            survival_time, event, target_binary = self.targets.iloc[idx]

            survival = encode_survival(survival_time, event, self.time_bins)
        else:
            survival_time = event = target_binary = survival = -1

        emr = self.emr_data.iloc[idx]
        emr = torch.tensor(emr.values, dtype=torch.float)

        if self.emr_transform is not None:
            emr = self.emr_transform(emr)

        # load image and GTV mask
        path = os.path.join(self.root_directory, self.split,
                            "{}", f"{subject_id}.nrrd")
        image = sitk.ReadImage(path.format("images"))
        mask = sitk.ReadImage(path.format("masks"))

        image = sitk.Clamp(image, sitk.sitkFloat32, -500, 1000)

        if self.transform is not None:
            # transform might return multiple images
            # if the two-stream model is used
            *image, mask = self.transform([image, mask])

        return Data(
            image=image,
            emr=emr,
            survival=survival,
            survival_time=survival_time,
            event=event,
            target_binary=target_binary,
            mask=mask)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.emr_data)
