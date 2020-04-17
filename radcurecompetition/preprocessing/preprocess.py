import os
from argparse import ArgumentParser

import pandas as pd

from imgtools.pipeline import Pipeline
from imgtools.ops import ImageFileInput, ImageFileOutput, MetadataOutput, StructureSetToSegmentation
from imgtools.io import read_dicom_series, read_dicom_rtstruct


class RadcurePipeline(Pipeline):
    def __init__(self,
                 input_directory,
                 output_directory,
                 id_mapping_path,
                 roi_names=["GTV"],
                 n_jobs=-1,
                 show_progress=True):

        super().__init__(n_jobs=n_jobs, show_progress=show_progress)

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.id_mapping_path = id_mapping_path
        self.roi_names = roi_names

        self.image_input = ImageFileInput(
            self.input_directory,
            get_subject_id_from="subject_directory",
            subdir_path="*/ImageSet_*",
            reader=read_dicom_series)
        self.structure_set_input = ImageFileInput(
            self.input_directory,
            get_subject_id_from="subject_directory",
            subdir_path="*/structures/RTSTRUCT.dcm",
            reader=read_dicom_rtstruct)

        self.make_binary_mask = StructureSetToSegmentation(
            roi_names=self.roi_names)

        self.image_output = ImageFileOutput(
            os.path.join(self.output_directory, "images"),
            filename_format="{subject_id}.nrrd",
            create_dirs=True,
            compress=True)
        self.mask_output = ImageFileOutput(os.path.join(self.output_directory, "masks"),
                                           filename_format="{subject_id.nrrd",
                                           create_dirs=True,
                                           compress=True)

        self.id_mapping = pd.read_csv(self.id_mapping_path,
                                      usecols=["MRN, Study ID"],
                                      index_col="MRN")

    def process_one_subject(self, subject_id):
        image, rtstruct = self.image_input(subject_id), self.image_input(subject_id)
        mask = self.make_binary_mask(rtstruct, image)

        new_subject_id = self.id_mapping.loc[subject_id, "Study ID"]
        self.image_output(new_subject_id, image)
        self.mask_output(new_subject_id, mask)


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
        help="Path to CSV file with a 'MRN' column containing the original patient IDs and 'Study ID' column containing the corresponding de-identified IDs."
    )
    parser.add_argument("--roi_names",
                        nargs="*",
                        default=["GTV"],
                        help="List of ROI names to extract.")
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
                               n_jobs=args.n_jobs,
                               show_progress=args.show_progress)
    pipeline.run()


if __name__ == "__main__":
    main()
