import os
from argparse import ArgumentParser
from typing import List, Optional

import numpy as np
import SimpleITK as sitk

from imgtools.pipeline import Pipeline
from imgtools.ops import ImageFileInput, ImageFileOutput, Centroid
from imgtools.ops.functional import crop


class PreprocessingPipeline(Pipeline):
    def __init__(self,
                 input_directory: str,
                 output_directory: str,
                 crop_size=[130, 130, 70],
                 neck_crop_size=[160, 160, 110],
                 n_jobs: int = -1,
                 show_progress: bool = True):
        super().__init__(n_jobs=n_jobs, show_progress=show_progress, warn_on_error=True)

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.crop_size = crop_size
        self.neck_crop_size = neck_crop_size

        self.image_input = ImageFileInput(
            os.path.join(self.input_directory, "images"),
            get_subject_id_from="filename")
        self.mask_input = ImageFileInput(
            os.path.join(self.input_directory, "masks"),
            get_subject_id_from="filename")

        self.find_centroid = Centroid()

        self.image_output = ImageFileOutput(
            os.path.join(self.output_directory, "images"),
            filename_format="{subject_id}.nrrd",
            create_dirs=True,
            compress=True)
        self.neck_image_output = ImageFileOutput(
            os.path.join(self.output_directory, "images"),
            filename_format="{subject_id}_neck.nrrd",
            create_dirs=True,
            compress=True)
        self.mask_output = ImageFileOutput(
            os.path.join(self.output_directory, "masks"),
            filename_format="{subject_id}.nrrd",
            create_dirs=True,
            compress=True)


    def process_one_subject(self, subject_id: str):
        """Preprocess the image and segmentation of one subject.

        Parameters
        ----------
        subject_id
            The ID of the currently processed subject."""
        image = self.image_input(subject_id)
        tumour_mask = self.mask_input(subject_id)

        tumour_centroid = self.find_centroid(tumour_mask)

        cropped_tumour = crop(image, tumour_centroid, self.crop_size)
        cropped_mask = crop(tumour_mask, tumour_centroid, self.crop_size)

        patient_centroid_z = image.GetSize()[2] // 2
        patient_mask_slice = sitk.BinaryErode(image[:, :, patient_centroid_z] > -800, 3)
        patient_mask_slice = sitk.RelabelComponent(sitk.ConnectedComponent(patient_mask_slice))
        patient_centroid_xy = self.find_centroid(patient_mask_slice == 1)

        patient_centroid = [
            patient_centroid_xy[0],
            patient_centroid_xy[1],
            patient_centroid_z
        ]

        cropped_neck = crop(image, patient_centroid, self.neck_crop_size)

        self.image_output(subject_id, cropped_tumour)
        self.neck_image_output(subject_id, cropped_neck)
        self.mask_output(subject_id, cropped_mask)


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
    parser.add_argument("--crop_size",
                        type=int,
                        nargs=3,
                        default=[130, 130, 70])
    parser.add_argument("--neck_crop_size",
                        type=int,
                        nargs=3,
                        default=[160, 160, 110])
    parser.add_argument("--n_jobs",
                        type=int,
                        default=-1,
                        help="Number of parallel processes to use.")
    parser.add_argument("--show_progress",
                        action="store_true",
                        help="Print progress to stdout.")

    args = parser.parse_args()
    pipeline = PreprocessingPipeline(args.input_directory,
                                     args.output_directory,
                                     crop_size=args.crop_size,
                                     neck_crop_size=args.neck_crop_size,
                                     n_jobs=args.n_jobs,
                                     show_progress=args.show_progress)
    pipeline.run()


if __name__ == "__main__":
    main()
