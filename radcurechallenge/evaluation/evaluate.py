import os
from argparse import ArgumentParser
from hashlib import md5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .metrics import (evaluate_binary, evaluate_survival,
                      plot_roc_curve, plot_pr_curve)


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = pd.read_csv(args.true_data_path, index_col="Study ID")
    targets = data.loc[data["split"] == "test", ["target_binary", "death", "survival_time"]]
    targets = targets.sort_index()
    volume = data.loc[data["split"] == "test", "volume"].sort_index()

    files = filter(lambda x: x.name.endswith(".csv") and not x.name.startswith("excluded"), os.scandir(args.predictions_dir))
    all_predictions = []
    for f in files:
        group, team, name = os.path.splitext(f.name)[0].split("_")
        predictions = pd.read_csv(f.path, index_col="Study ID").sort_index()

        assert (targets.index == predictions.index).all()

        all_predictions.append({"group": group, "team": team, "name": name, "predictions": predictions})

    for name, team in [(p["name"], p["team"]) for p in all_predictions if p["group"] == "challenge"]:
        challenge_predictions = pd.concat([p["predictions"].reset_index() for p in all_predictions if p["team"] != team and p["name"] != name])
        ensemble_predictions = challenge_predictions.groupby("Study ID").mean()
        all_predictions.append({"group": "challenge", "team": f"ensemble \ {team}-{name}", "name": "combined", "predictions": ensemble_predictions})
    challenge_predictions = pd.concat([p["predictions"].reset_index() for p in all_predictions if p["group"] == "challenge"])
    ensemble_predictions = challenge_predictions.groupby("Study ID").mean()
    all_predictions.append({"group": "challenge", "team": "ensemble", "name": "combined", "predictions": ensemble_predictions})

    results = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for p in all_predictions:
        group, team, name, predictions = p["group"], p["team"], p["name"], p["predictions"]

        submission_id = md5(("team" + "name").encode()).hexdigest()[:7]

        cur_res = {"group": group, "team": team, "name": name, "submission_id": submission_id}
        if "binary" in predictions:
            metrics_binary = evaluate_binary(targets["target_binary"],
                                             predictions["binary"],
                                             n_permutations=args.n_permutations,
                                             n_jobs=args.n_jobs)
            cur_res.update(metrics_binary)
            ax[0] = plot_roc_curve(targets["target_binary"],
                                   predictions["binary"],
                                   label=f"{group}-{team}-{name}",
                                   ax=ax[0])
            ax[1] = plot_pr_curve(targets["target_binary"],
                                  predictions["binary"],
                                  label=f"{group}-{team}-{name}",
                                  ax=ax[1])
            volume_corr, volume_corr_pval = spearmanr(predictions["binary"], volume)
            cur_res["volume_corr"] = volume_corr
            cur_res["volume_corr_pval"] = volume_corr_pval

        if "survival_event" not in predictions:
            predictions["survival_event"] = predictions["binary"]

        if "survival_time_0" in predictions:
            time_pred = np.array(predictions.filter(like="survival_time").values)
        else:
            time_pred = None
        metrics_survival = evaluate_survival(targets["death"],
                                             targets["survival_time"],
                                             predictions["survival_event"],
                                             time_pred,
                                             n_permutations=args.n_permutations,
                                             n_jobs=args.n_jobs)
        cur_res.update(metrics_survival)
        results.append(cur_res)

    ax[0].legend()
    ax[1].legend()

    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    fig.savefig(os.path.join(args.output_dir, "roc_pr_curves.png"), dpi=300)


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
    parser.add_argument("--n_permutations", type=int, default=5000,
                        help=("How many random permutations to use when"
                              "evaluating significance."))
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes to use.")
    args = parser.parse_args()
    main(args)
