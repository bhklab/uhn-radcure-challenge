import os
import glob
from argparse import ArgumentParser
from typing import List, Optional

import numpy as np
import pandas as pd

from imgtools.pipeline import Pipeline
from imgtools.ops import ImageCSVInput, ImageFileOutput, MetadataOutput, StructureSetToSegmentation
from imgtools.io import read_dicom_series, read_dicom_rtstruct


class RadcurePipeline(Pipeline):
    """This pipeline can be used to preprocess and anonymize the images and
    clinical data using the RADCURE challenge specifications. Specifically the
    pipeline can:
        - automatically exclude ineligible patients
        - select and clean up the clinical variables
        - convert images from DICOM to NRRD
        - generate primary GTV binary masks and save in NRRD format
        - split the dataset into training and validation subsets.
    """
    def __init__(self,
                 input_directory: str,
                 output_directory: str,
                 clinical_data_path: str,
                 roi_names: List[str] = ["GTV"],
                 train_size: float = .7,
                 save_clinical_path: Optional[str] = "../../data/splits.csv",
                 n_jobs: int = -1,
                 show_progress: bool = True):
        """Initialize the pipeline.

        Parameters
        ----------
        input_directory
            Path to directory containing the RADCURE images.

        output_directory
            Path to directory where the preprocessed images will be saved.

        clinical_data_path
            Path to CSV file containing RADCURE clinical data.

        roi_names
            List of ROI names to convert to binary masks.

        train_size
            The fraction of the dataset to use for training.

        save_clinical_path
            Path where the preprocessed clinical data will be saved.

        n_jobs
            Number of parallel processes to use.

        show_progress
            Print progress updates to stdout.
        """
        super().__init__(n_jobs=n_jobs, show_progress=show_progress)

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.clinical_data_path = clinical_data_path
        self.roi_names = roi_names
        self.train_size = train_size
        self.save_clinical_path = save_clinical_path

        self.clinical_data = self.load_clinical_data()
        if self.save_clinical_path:
            self.clinical_data.drop(["image_path", "rtstruct_path"], axis=1).to_csv(self.save_clinical_path, index=False)

        # self.image_input = ImageFileInput(
        #     self.input_directory,
        #     get_subject_id_from="subject_directory",
        #     subdir_path="*/ImageSet_*",
        #     reader=read_dicom_series)
        # self.structure_set_input = ImageFileInput(
        #     self.input_directory,
        #     get_subject_id_from="subject_directory",
        #     subdir_path="*/structures/RTSTRUCT.dcm",
        #     reader=read_dicom_rtstruct)

        self.image_input = ImageCSVInput(self.clinical_data,
                                         colnames=["image_path", "rtstruct_path"],
                                         id_column="Study ID",
                                         readers=[read_dicom_series, read_dicom_rtstruct])

        self.make_binary_mask = StructureSetToSegmentation(
            roi_names=self.roi_names)

        self.image_output = ImageFileOutput(
            self.output_directory,
            filename_format="{split}/images/{subject_id}.nrrd",
            create_dirs=True,
            compress=True)
        self.mask_output = ImageFileOutput(
            self.output_directory,
            filename_format="{split}/masks/{subject_id}.nrrd",
            create_dirs=True,
            compress=True)

        self.clinical_data = self.clinical_data.set_index("Study ID")

    def load_clinical_data(self) -> pd.DataFrame:
        """Load and preprocess the clinical data.

        This method will exclude cases not meeting the inclusion criteria, find
        image and RTSTRUCT paths, clean up the clinical variables and split the
        data into training and validation subsets.

        Returns
        -------
        pd.DataFrame
            The processed clinical data."""
        data = pd.read_csv(self.clinical_data_path)

        sites = [
            "Oropharynx",
            "Larynx",
            "Nasopharynx",
            "Hypopharynx",
            "Lip & Oral Cavity",
            "Paranasal Sinus",
            "nasal cavity",
            "Nasal Cavity",
            "esophagus",
            "Esophagus",
            "Salivary Glands"
        ]


        image_paths = []
        rtstruct_paths = []
        for mrn in data["MRN"]:
            try:
                image_path = glob.glob(os.path.join(self.input_directory, str(mrn), "*", "ImageSet*"))[0]
            except IndexError:
                image_path = None

            try:
                rtstruct_path = glob.glob(os.path.join(self.input_directory, str(mrn), "*", "structures", "RTSTRUCT*"))[0]
            except IndexError:
                rtstruct_path = None

            image_paths.append(image_path)
            rtstruct_paths.append(rtstruct_path)

        data["image_path"] = image_paths
        data["rtstruct_path"] = rtstruct_paths
        data["has_image"] = (~data["image_path"].isnull()) & (~data["rtstruct_path"].isnull())

        # exclusion criteria:
        # - missing image or RTSTRUCT
        # - site other than oro/naso/hypopharynx or oral cavity
        # - < 2 yr follow-up
        # - M stage > 0
        # - prior surgery
        data = data[(
            data["has_image"] &
            data["Disease Site"].isin(sites) &
            ((data["Length FU"] >= 2) | (data["Status Last Follow Up"] == "Dead")) &
            (data["M Stage"] == "M0") &
            (data["Tx Intent"] != "post-op")
        )]

        data = self.split_by_date(data)

        # create target columns
        data["target_binary"] = np.uint8((data["Length FU"] <= 2) & (data["Status Last Follow Up"] == "Dead"))
        data["survival_time"] = data["Length FU"]
        data["death"] = data["Status Last Follow Up"].map({"Dead": 1, "Alive": 0})

        # clean up the clinical variables for phase II
        data["Chemotherapy"] = data["Chemotherapy"].map(lambda x: 0 if "none" in x.lower() else 1, na_action="ignore")
        data["Disease Site"] = data["Disease Site"].str.lower()
        data["HPV Combined"] = data["HPV Combined"].map({"Yes, positive": 1, "Yes, Negative": 0})
        data["Stage"] = data["Stage-7th"]

        data = data[[
            "Study ID",
            "split",
            "image_path",
            "rtstruct_path",
            "target_binary",
            "survival_time",
            "death",
            "age at dx",
            "Sex",
            "T Stage",
            "N Stage",
            "Stage",
            "Dose",
            "Chemotherapy",
            "HPV Combined",
            "Disease Site"
        ]]

        return data


    def split_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Split the dataset into training and validation subsets.

        This method will add a new column named 'split' containing the subset
        membership for each subject. Splitting is performed by date of
        diagnosis. The split date is chosen so that at least `self.train_size`
        subjects are in the training set.

        Returns
        -------
        pd.DataFrame
            The updated dataset.
        """
        data["Date of dx"] = pd.to_datetime(data["Date of dx"])
        cases_per_month = data.set_index("Date of dx").groupby(pd.Grouper(freq="M")).size()

        # compute the cumulative % of total cases in each month
        cumulative_freqs = cases_per_month.cumsum() / cases_per_month.sum()
        # find the lowest date with >= train_size % data
        split_date = cumulative_freqs[cumulative_freqs >= self.train_size].index[0]

        data["split"] = np.where(data["Date of dx"] <= split_date, "training", "validation")

        return data


    def process_one_subject(self, subject_id: str):
        """Preprocess the image and segmentation of one subject.

        Parameters
        ----------
        subject_id
            The ID of the currently processed subject."""
        image, rtstruct = self.image_input(subject_id)
        mask = self.make_binary_mask(rtstruct, image)

        split = self.clinical_data.loc[subject_id, "split"]
        self.image_output(subject_id, image, split=split)
        self.mask_output(subject_id, mask, split=split)


def main():
    parser = ArgumentParser(description="Split RADCURE images into train/test sets and save in NRRD format.")
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to directory containing original RADCURE images.")
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path to directory where the processed dataset will be saved.")
    parser.add_argument(
        "clinical_data_path",
        type=str,
        help="Path to CSV file with clinical data.")
    parser.add_argument("--roi_names",
                        nargs="*",
                        default=["GTV"],
                        help="List of ROI names to extract.")
    parser.add_argument("--train_size",
                        type=float,
                        default=.7,
                        help="Fraction of data to use as the training set.")
    parser.add_argument(
        "--save_clinical_path",
        type=str,
        default="../../data/splits.csv",
        help="Path to CSV file where the training/test split information for each subject will be saved.")
    parser.add_argument("--n_jobs",
                        type=int,
                        default=-1,
                        help="Number of parallel processes to use.")
    parser.add_argument("--show_progress",
                        action="store_true",
                        help="Print progress to stdout.")

    args = parser.parse_args()
    pipeline = RadcurePipeline(args.input_directory,
                               args.output_directory,
                               args.clinical_data_path,
                               roi_names=args.roi_names,
                               train_size=args.train_size,
                               save_clinical_path=args.save_clinical_path,
                               n_jobs=args.n_jobs,
                               show_progress=args.show_progress)
    pipeline.run()


if __name__ == "__main__":
    main()
