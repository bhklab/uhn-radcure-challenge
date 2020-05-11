"""Extract PyRadiomics features from the training and test datasets."""
import os
import tempfile
import subprocess
from argparse import ArgumentParser

import pandas as pd


def extract_features(args):
    data = pd.read_csv(args.id_split_path)[["Study ID", "split", "target_binary", "survival_time", "death"]]
    data["Image"] = data.apply(lambda x: os.path.join(
        args.image_directory, x["split"], "images", x["Study ID"] + ".nrrd"), axis=1)
    data["Mask"] = data.apply(lambda x: os.path.join(
        args.image_directory, x["split"], "masks", x["Study ID"] + ".nrrd"), axis=1)
    tmp_path = os.path.join(tempfile.gettempdir(), "radiomics.csv")
    data.to_csv(tmp_path, index=False) # this is necessary since pyradiomics won't
                                       # overwrite an existing file for some reason
    # using the pyradiomics batch command is easier than setting up the
    # feature extractor object
    command = [
        "pyradiomics",
        tmp_path,
        "-o", args.output_path,
        "-f", "csv",
        "--jobs", str(args.n_jobs),
        "--param", args.params_file
    ]
    subprocess.run(command)
    os.remove(tmp_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image_directory",
                        type=str,
                        help="Path to directory containing images and segmentation masks.")
    parser.add_argument("id_split_path",
                        type=str,
                        help="Path to CSV file containing subject IDs and split information.")
    parser.add_argument("--output_path", "-o",
                        type=str,
                        help="Path where the extracted features will be saved.")
    parser.add_argument("--params_file",
                        type=str,
                        help="Path to PyRadiomics configuration file.")
    parser.add_argument("--n_jobs",
                        type=int,
                        default=1,
                        help="Number of parallel processes to use.")
    args = parser.parse_args()
    extract_features(args)
