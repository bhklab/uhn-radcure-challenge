import glob
import os
import subprocess
from argparse import ArgumentParser

import pandas as pd


def extract_features(image_directory, id_split_path, output_path, params_file, n_jobs=1):
    data = pd.read_csv(id_split_path)[["Study ID", "split"]]
    data["Image"] = data.apply(lambda x: os.path.join(
        image_directory, x["split"], "images", x["Study ID"] + ".nrrd"), axis=1)
    data["Mask"] = data.apply(lambda x: os.path.join(
        image_directory, x["split"], "masks", x["Study ID"] + ".nrrd"), axis=1)
    data.to_csv(output_path)
    command = [
        "pyradiomics",
        output_path,
        "-o", output_path,
        "-f", "csv",
        "--jobs", str(n_jobs),
        "--param", params_file
    ]
    subprocess.run(command)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image_directory",
                        type=str,
                        help="Path to directory containing images and segmentation masks.")
    parser.add_argument("id_split_data_path",
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
    extract_features(args.image_directory, args.id_split_path,
                     args.output_path, args.params_file, args.n_jobs)
