from argparse import ArgumentParser

import pandas as pd

from ...evaluation import evaluate_binary, evaluate_survival
from .base import SimpleBaseline


def main(args):
    clinical_columns = [
        "age at dx",
        "Sex",
        "T Stage",
        "N Stage",
        "HPV Combined",
        "target_binary",
        "survival_time",
        "death",
    ]
    radiomics_columns = "firstorder|shape|glcm|glszm|glrlm|gldm|ngtdm"

    clinical = pd.read_csv(args.clinical_data_path)
    clinical["T Stage"] = clinical["T stage"].map({
        "T1": "T1/2",
        "T1a": "T1/2",
        "T1b": "T1/2",
        "T2": "T1/2",
        "T2 (2)": "T1/2",
        "T3": "T3/4",
        "T3 (2)": "T3/4",
        "T4": "T3/4",
        "T4a": "T3/4",
        "T4b": "T3/4"
    })
    clinical["N Stage"] = clinical["N stage"].map({
        "N0": "N0",
        "N1": "N1",
        "N2": "N2",
        "N2a": "N2",
        "N2b": "N2",
        "N2c": "N2",
        "N3": "N3",
        "N3a": "N3",
        "N3b": "N3"
    })

    radiomics = pd.read_csv(args.radiomics_data_path)

    baselines = {
        "clinical_baseline": SimpleBaseline(clinical,
                                            max_features_to_select=0,
                                            colnames=clinical_columns),
        "radiomics_baseline": SimpleBaseline(radiomics,
                                             max_features_to_select=args.max_features_to_select,
                                             colnames=radiomics_columns),
        "volume_baseline": SimpleBaseline(radiomics,
                                          max_features_to_select=0,
                                          colnames=["original_shape_MeshVolume"])
    }

    results = []
    for name, baseline in baselines.items():
        true, pred = baseline.get_test_predictions()
        results_binary = evaluate_binary(true["binary"],
                                         pred["binary"],
                                         args.n_permutations,
                                         args.n_jobs)
        results_survival = evaluate_survival(true["survival_event"],
                                             true["survival_time"],
                                             pred["survival_event"],
                                             pred["survival_time"],
                                             args.n_permutations,
                                             args.n_jobs)
        results.append({"name": name, **results_binary, **results_survival})

    results = pd.DataFrame(results)
    results.to_csv(args.output_path, index=False)


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
