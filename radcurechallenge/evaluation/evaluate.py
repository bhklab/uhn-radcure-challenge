import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import evaluate_binary, evaluate_survival, plot_roc_curve, plot_pr_curve


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = pd.read_csv(args.true_data_path, index_col="Study ID")
    targets = data.loc[data["split"] == "test", ["target_binary", "death", "survival_time"]]
    targets = targets.sort_index()

    files = filter(lambda x: x.name.endswith(".csv"), os.scandir(args.predictions_dir))
    results = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for f in files:
        group, name = os.path.splitext(f.name)[0].split("_")
        predictions = pd.read_csv(f.path, index_col="Study ID").sort_index()

        assert (targets.index == predictions.index).all()

        cur_res = {"group": group, "name": name}
        if "binary" in predictions.columns:
            metrics_binary = evaluate_binary(targets["target_binary"],
                                             predictions["binary"],
                                             n_permutations=args.n_permutations,
                                             n_jobs=args.n_jobs)
            cur_res.update(metrics_binary)
            ax[0] = plot_roc_curve(targets["target_binary"],
                                   predictions["binary"],
                                   label=f"{group}-{name}",
                                   ax=ax[0])
            ax[1] = plot_pr_curve(targets["target_binary"],
                                  predictions["binary"],
                                  ax=ax[1])
        if "survival_time_0" in predictions.columns:
            time_pred = np.array(predictions.filter(like="survival_time").values)
            metrics_survival = evaluate_survival(targets["death"],
                                                 targets["survival_time"],
                                                 predictions["survival_event"],
                                                 time_pred,
                                                 n_permutations=args.n_permutations,
                                                 n_jobs=args.n_jobs)
            cur_res.update(metrics_survival)

        results.append(cur_res)

        fig.legend()

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
