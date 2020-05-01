from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve

from ...evaluation import evaluate_binary, evaluate_survival
from .base import SimpleBaseline


np.random.seed(42)


def main(args):
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
    clinical["T Stage"] = clinical["T Stage"].map({
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
    clinical["N Stage"] = clinical["N Stage"].map({
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

    clinical = pd.get_dummies(clinical,
                              columns=["Sex",
                                       "T Stage",
                                       "N Stage",
                                       "HPV Combined"],
                              drop_first=True)

    radiomics = pd.read_csv(args.radiomics_data_path)

    baselines = {
        "clinical_baseline": SimpleBaseline(clinical,
                                            max_features_to_select=0,
                                            colnames=clinical_columns,
                                            n_jobs=args.n_jobs),
        "radiomics_baseline": SimpleBaseline(radiomics,
                                             max_features_to_select=args.max_features_to_select,
                                             colnames=radiomics_columns,
                                             n_jobs=args.n_jobs),
        "volume_baseline": SimpleBaseline(radiomics,
                                          max_features_to_select=0,
                                          colnames=["original_shape_MeshVolume"],
                                          n_jobs=args.n_jobs)
    }

    results = []
    binary_predictions = []
    for name, baseline in baselines.items():
        true, pred = baseline.get_test_predictions()
        results_binary = evaluate_binary(true["binary"],
                                         pred["binary"],
                                         n_permutations=args.n_permutations,
                                         n_jobs=args.n_jobs)
        results_survival = evaluate_survival(true["survival_event"],
                                             true["survival_time"],
                                             pred["survival_event"],
                                             pred["survival_time"],
                                             n_permutations=args.n_permutations,
                                             n_jobs=args.n_jobs)
        results.append({"name": name, **results_binary, **results_survival})
        binary_predictions.append({"name": name, "true": true["binary"], "pred": pred["binary"][:, 1]})

    results = pd.DataFrame(results)
    results.to_csv(args.output_path, index=False)

    if args.plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        for p in binary_predictions:
            fpr, tpr, _ = roc_curve(p["true"], p["pred"])
            ax[0].plot([0, 1], [0, 1], c="grey", linestyle="--")
            ax[0].plot(fpr, tpr, label=p["name"].replace("_baseline", ""))
            ax[0].set_xlabel("False positive rate")
            ax[0].set_ylabel("True positive rate")
            ax[0].set_title("ROC curves")
            precision, recall, _ = precision_recall_curve(p["true"], p["pred"])
            ax[1].plot(recall, precision)
            ax[1].set_xlabel("Recall")
            ax[1].set_ylabel("Precision")
            ax[1].set_title("Precision-recall curves")
            np.save(f"pred_{p['name']}.npy", p["pred"], pred)
        fig.legend()
        fig.savefig(args.output_path.replace(".csv", ".png"), dpi=300)


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
    parser.add_argument("--plot", action="store_true",
                        help="Plot ROC and precision-recall curves for each model.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes to use.")

    args = parser.parse_args()
    main(args)
