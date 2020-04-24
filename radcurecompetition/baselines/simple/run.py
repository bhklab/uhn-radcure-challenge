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

    baselines = {
        "clinical_baseline": SimpleBaseline(args.clinical_data_path,
                                            max_features_to_select=0,
                                            colnames=clinical_columns),
        "radiomics_baseline": SimpleBaseline(args.radiomics_data_path,
                                             max_features_to_select=args.max_features_to_select,
                                             colnames=radiomics_columns),
        "volume_baseline": SimpleBaseline(args.radiomics_data_path,
                                          max_features_to_select=0,
                                          colnames=["original_shape_Volume"])  # XXX change to actual colname
    }

    results = []
    for name, baseline in baselines:
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
    results.to_csv(args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("clinical_data_path", type=str,
                        help="Path to CSV file with clinical data.")
    parser.add_argument("radiomics_data_path", type=str,
                        help="Path to CSV file with radiomics features.")
    parser.add_argument("radiomics_data_path", type=str,
                        help="Path to CSV file with radiomics features.")
    parser.add_argument("--output_path", "-o", type=str,
                        help="Path where the results will be saved.")
    parser.add_argument("--max_features_to_select", type=int, default=10,
                        help="Maximum number of features in the radiomics model.")

    args = parser.parse_args()
    main(args)
