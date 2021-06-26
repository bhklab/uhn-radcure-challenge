import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import evaluate_binary, evaluate_survival


np.random.seed(42)


def load_predictions(predictions_dir: str):
    files = filter(lambda x: x.name.endswith(".csv") and not x.name.startswith("excluded"),
                   os.scandir(predictions_dir))
    all_predictions = []
    for f in files:
        group, kind, submission_id = os.path.splitext(f.name)[0].split("_")
        if submission_id == 'ensemble':
            continue
        predictions = pd.read_csv(f.path, index_col="Study ID").sort_index()

        if "survival_event" not in predictions:
            predictions["survival_event"] = predictions["binary"]

        all_predictions.append({
            "group": group,
            "kind": kind,
            "submission_id": submission_id,
            "predictions": predictions
        })
    return all_predictions


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = pd.read_csv(args.true_data_path, index_col="Study ID")
    targets = data.loc[data["split"] == "test", ["target_binary", "death", "survival_time"]]
    targets = targets.sort_index()
    volume = data.loc[data["split"] == "test", "volume"].sort_index()

    all_predictions = load_predictions(args.predictions_dir)

    # create ensemble submission by averaging predictions
    challenge_predictions = pd.concat([
        p["predictions"].reset_index()
        for p in all_predictions if p["group"] == "challenge"
    ])
    ensemble_predictions = challenge_predictions.groupby("Study ID").mean()
    all_predictions.append({
        "group": "challenge",
        "kind": "combined",
        "submission_id": "ensemble",
        "predictions": ensemble_predictions
    })

    # compute metrics for all submissions (including ensemble)
    results = []
    for p in all_predictions:
        group, kind, submission_id, predictions = p["group"], p["kind"], p["submission_id"], p["predictions"]

        cur_res = {"group": group, "kind": kind, "submission_id": submission_id}
        if "binary" in predictions:
            metrics_binary = evaluate_binary(targets["target_binary"].values,
                                             predictions["binary"].values,
                                             n_permutations=args.n_permutations,
                                             n_jobs=args.n_jobs)
            cur_res.update(metrics_binary)
            volume_corr, volume_corr_pval = spearmanr(predictions["binary"], volume)
            cur_res["volume_corr"] = volume_corr
            cur_res["volume_corr_pval"] = volume_corr_pval

        if "survival_time_0" in predictions:
            time_pred = np.array(predictions.filter(like="survival_time").values)
        else:
            time_pred = None
        metrics_survival = evaluate_survival(targets["death"].values,
                                             targets["survival_time"].values,
                                             predictions["survival_event"].values,
                                             time_pred,
                                             n_permutations=args.n_permutations,
                                             n_jobs=args.n_jobs)
        cur_res.update(metrics_survival)
        results.append(cur_res)

    # save ensemble predictions and results
    ensemble_predictions.to_csv(os.path.join(args.predictions_dir, "challenge_combined_ensemble.csv"))
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("true_data_path", type=str,
                        help="Path to CSV files with ground truth data.")
    parser.add_argument("predictions_dir", type=str,
                        help=("Path to directory containing CSV files with"
                              "model predictions."))
    parser.add_argument("output_dir", type=str,
                        help=("Path to directory where the computed metrics"
                              "and figures will be saved."))
    parser.add_argument("--n_permutations", type=int, default=10000,
                        help=("How many random permutations to use when"
                              "evaluating significance."))
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes to use.")
    args = parser.parse_args()
    main(args)
