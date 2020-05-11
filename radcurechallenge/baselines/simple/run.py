import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from .base import SimpleBaseline


np.random.seed(42)


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    clinical_columns = [
        "age at dx",
        "Sex_Male",
        "T Stage_T3/4",
        "N Stage_N1",
        "N Stage_N2",
        "N Stage_N3",
        "HPV Combined_1.0",
        "target_binary",
        "survival_time",
        "death",
    ]
    radiomics_columns = "firstorder|shape|glcm|glszm|glrlm|gldm|ngtdm"

    clinical = pd.read_csv(args.clinical_data_path)
    # binarize T stage as T1/2 = 0, T3/4 = 1
    clinical["T Stage"] = clinical["T Stage"].map({
        "T1":     "T1/2",
        "T1a":    "T1/2",
        "T1b":    "T1/2",
        "T2":     "T1/2",
        "T2 (2)": "T1/2",
        "T3":     "T3/4",
        "T3 (2)": "T3/4",
        "T4":     "T3/4",
        "T4a":    "T3/4",
        "T4b":    "T3/4"
    })
    # use more fine-grained grouping for N stage
    clinical["N Stage"] = clinical["N Stage"].map({
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

    clinical = pd.get_dummies(clinical,
                              columns=["Sex",
                                       "T Stage",
                                       "N Stage",
                                       "HPV Combined"],
                              drop_first=True)

    radiomics = pd.read_csv(args.radiomics_data_path)

    baselines = {
        "clinical": SimpleBaseline(clinical,
                                   max_features_to_select=0,
                                   colnames=clinical_columns,
                                   n_jobs=args.n_jobs),
        "radiomics": SimpleBaseline(radiomics,
                                    max_features_to_select=args.max_features_to_select,
                                    colnames=radiomics_columns,
                                    n_jobs=args.n_jobs),
        "volume": SimpleBaseline(radiomics,
                                 max_features_to_select=0,
                                 colnames=["original_shape_MeshVolume"],
                                 n_jobs=args.n_jobs)
    }

    test_ids = clinical.loc[clinical["split"] == "test", "Study ID"]
    for name, baseline in baselines.items():
        pred = baseline.get_test_predictions()
        survival_time = pred.pop("survival_time")
        for i, col in enumerate(survival_time.T):
            pred[f"survival_time_{i}"] = col

        out_path = os.path.join(args.output_path, f"baseline_{name}.csv")
        pd.DataFrame(pred, index=test_ids).to_csv(out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("clinical_data_path", type=str,
                        help="Path to CSV file with clinical data.")
    parser.add_argument("radiomics_data_path", type=str,
                        help="Path to CSV file with radiomics features.")
    parser.add_argument("--output_path", "-o", type=str,
                        help="Path where the results will be saved.")
    parser.add_argument("--max_features_to_select", type=int, default=10,
                        help="Maximum number of features in the radiomics model.")
    parser.add_argument("--n_permutations", type=int, default=5000,
                        help="How many random permutations to use when evaluating significance.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes to use.")

    args = parser.parse_args()
    main(args)
