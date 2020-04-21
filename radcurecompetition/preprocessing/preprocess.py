import os
import glob
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from imgtools.pipeline import Pipeline
from imgtools.ops import ImageCSVInput, ImageFileOutput, MetadataOutput, StructureSetToSegmentation
from imgtools.io import read_dicom_series, read_dicom_rtstruct


class RadcurePipeline(Pipeline):
    def __init__(self,
                 input_directory,
                 output_directory,
                 clinical_data_path,
                 roi_names=["GTV"],
                 train_size=.7,
                 save_splits_path="../../data/splits.csv",
                 n_jobs=-1,
                 show_progress=True):

        super().__init__(n_jobs=n_jobs, show_progress=show_progress)

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.clinical_data_path = clinical_data_path
        self.roi_names = roi_names
        self.train_size = train_size
        self.save_splits_path = save_splits_path

        self.clinical_data = self.load_clinical_data()
        if self.save_splits_path:
            self.clinical_data.drop("MRN", axis=1).to_csv(self.save_splits_path, index=False)

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
            os.path.join(self.output_directory, "images"),
            filename_format="{split}/{subject_id}.nrrd",
            create_dirs=True,
            compress=True)
        self.mask_output = ImageFileOutput(
            os.path.join(self.output_directory, "masks"),
            filename_format="{split}/{subject_id}.nrrd",
            create_dirs=True,
            compress=True)

        self.clinical_data = self.clinical_data.set_index("MRN")

    def load_clinical_data(self):
        data = pd.read_csv(self.clinical_data_path)
        images = os.listdir(self.input_directory)
        rtstructs = [i for i in images if glob.glob(os.path.join(self.input_directory, i, "*", "*", "RTSTRUCT*"))]

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

        data["has_image"] = (data["MRN"].isin(images)) & (data["MRN"].isin(rtstructs))

        data["image_path"] = [glob.glob(os.path.join(self.input_directory, mrn, "*", "ImageSet*")) for mrn in data["MRN"]]
        data["rtstruct_path"] = [glob.glob(os.path.join(self.input_directory, mrn, "*", "structures", "RTSTRUCT*")) for mrn in data["MRN"]]

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
            (data["Tx Intent"] != "post-op") &
            ~(data["Date of death"].isnull())
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

        data = data[[
            "MRN",
            "Study ID",
            "image_path",
            "rtstruct_path",
            "target_binary",
            "survival_time",
            "death",
            "age at dx",
            "sex",
            "T stage",
            "N stage",
            "Stage",
            "Dose",
            "Chemotherapy",
            "HPV Combined",
            "Disease Site"
        ]]

        return data


    def split_by_date(self, data):
        data["Date of dx"] = pd.to_datetime(data["Date of dx"])
        cases_per_month = data.set_index("Date of dx").groupby(pd.Grouper(freq="M")).size()

        # compute the cumulative % of total cases in each month
        cumulative_freqs = cases_per_month.cumsum() / cases_per_month.sum()
        # find the lowest date with >= train_size % data
        split_date = cumulative_freqs[cumulative_freqs >= self.train_size].index[0]

        data["split"] = np.where(data["Date of dx"] < split_date, "train", "test")

        return data


    def process_one_subject(self, subject_id):
        image, rtstruct = self.image_input(subject_id)
        mask = self.make_binary_mask(rtstruct, image)

        new_subject_id = self.clinical_data.loc[subject_id, "Study ID"]
        split = self.clinical_data.loc[subject_id, "split"]
        self.image_output(new_subject_id, image, split=split)
        self.mask_output(new_subject_id, mask, split=split)


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
        "id_mapping_path",
        type=str,
        help="Path to CSV file with a 'MRN' column containing the original patient IDs and 'Study ID' column containing the corresponding de-identified IDs.")
    parser.add_argument("--roi_names",
                        nargs="*",
                        default=["GTV"],
                        help="List of ROI names to extract.")
    parser.add_argument("--train_size",
                        type=float,
                        default=.7,
                        help="Fraction of data to use as the training set.")
    parser.add_argument(
        "--save_splits_path",
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
                               args.id_mapping_path,
                               roi_names=args.roi_names,
                               train_size=args.train_size,
                               save_splits_path=args.save_splits_path,
                               n_jobs=args.n_jobs,
                               show_progress=args.show_progress)
    pipeline.run()


if __name__ == "__main__":
    main()
