import json
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
from scipy.stats.distributions import uniform, loguniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from tqdm import tqdm

from .mtlr import train_mtlr, mtlr_survival, mtlr_survival_at_times, mtlr_risk
from .utils import normalize, make_time_bins, make_data


torch.manual_seed(42)
np.random.seed(42)


def hyperparam_search(data_train, time_bins, param_dists, num_samples,
                      cv=5, verbose=True, **kwargs):
    """Optimizes MTLR hyperparameters using random search with cross-validation.

    Parameter
    ---------
    data_train : pd.DataFrame
        Dataframe with training features and targets.
    time_bins : np.ndarray
        The time bins to use for MTLR training.
    param_dists: dict
        Hyperparameter distributions to sample from. See the scikit-learn
        ParameterSampler documentation for allowed types.
    num_samples : int
        Number of random hyperparam samples to draw.
    cv : int
        Number of cross-validation folds to use (default 5).
    verbose : bool
        Whether to show progress bars and print other information during
        training.
    kwargs
        Additional keyword arguments passed to `train_mtlr`.

    Returns
    -------
    tuple of (mtlr.MTLR, dict)
        The trained best model and best hyperparameter settings found.
    """
    sampler = ParameterSampler(param_dists, num_samples)
    kfold = StratifiedKFold(n_splits=cv)
    auc_vals = []
    pbar = tqdm(sampler, total=num_samples,
                desc="finding best hparams",
                disable=not verbose)
    for hparams in pbar:
        fold_auc_vals = []
        for train_idx, val_idx in kfold.split(data_train, data_train["target_binary"]):
            train_fold, val_fold = data_train.iloc[train_idx], data_train.iloc[val_idx]
            x_val = torch.tensor(
                val_fold.drop(["time", "event", "target_binary"], axis=1).values,
                dtype=torch.float)
            model = train_mtlr(
                train_fold.drop("target_binary", axis=1), time_bins,
                verbose=False, **hparams, **kwargs)
            with torch.no_grad():
                pred = model(x_val)
                # compute AUC for predictions at 2 years
                pred_survival = mtlr_survival_at_times(pred, np.pad(time_bins, (1, 0)), np.array([24]))
            pred_binary = 1 - pred_survival[:, 0]
            try:
                val_auc = roc_auc_score(val_fold["target_binary"], pred_binary)
            except ValueError:
                val_auc = np.nan
            fold_auc_vals.append(val_auc)
        mean_fold_auc = sum(fold_auc_vals) / len(fold_auc_vals)
        auc_vals.append({"hparams": hparams, "val_auc": mean_fold_auc})

    best_run = max(auc_vals, key=lambda x: x["val_auc"])
    best_hparams, best_auc = best_run["hparams"], best_run["val_auc"]
    if verbose:
        print(f"best CV AUC = {best_auc:.3f}")
        print(f"training with hparams: {best_hparams}")

    model = train_mtlr(data_train.drop("target_binary", axis=1),
                       time_bins, verbose=verbose,
                       **best_hparams, **kwargs)
    return model, best_hparams


def make_test_predictions(model, data_test, time_bins, out_path):
    """Generates challenge predictions and saves them to file.

    Parameters
    ----------
    model : mtlr.MTLR
        Trained MTLR model.
    data_test : pd.DataFrame
        Dataframe with test features.
    time_bins : np.ndarray
        The time bins to use for MTLR training.
    out_path : str
        Path to CSV file where the predictions will be saved.
    """
    eval_times = np.linspace(0, 23, 24)
    with torch.no_grad():
        pred = model(torch.tensor(data_test.drop(["time", "event", "target_binary"], axis=1).values, dtype=torch.float))
        pred_binary = 1 - mtlr_survival(pred)[:, np.digitize(24, time_bins)]
        pred_survival = mtlr_survival_at_times(pred, np.pad(time_bins, (1, 0)), eval_times)
        pred_risk = mtlr_risk(pred).numpy()

    res = pd.DataFrame({
        "Study ID": data_test.index,
        "binary": pred_binary,
        "survival_event": pred_risk,
        **{f"survival_time_{i}": pred_survival[:, i+1] for i in range(0, 23)}
    })
    res.to_csv(out_path, index=False)


# ranges for LR and dropout were selected based on previous experiments
PARAM_DISTS = {
    "hidden_sizes": [[32], [64], [128],
                     [32]*2, [64]*2, [128]*2,
                     [32]*3, [64]*3, [128]*3],
    "dropout": uniform(0., .6),
    "lr": uniform(.001, .01),
    "batch_size": [128, 256, 512, 1024],
    "l2_reg": np.logspace(-2, 3, 6),
    "weight_decay": loguniform(1e-6, 1e-3),
}

def main(args):
    # prepare data
    data_train = make_data(args.clinical_data_path, split="training")
    if args.test_data_path:
        data_test = make_data(args.test_data_path)
    else:
        data_test = make_data(args.clinical_data_path, split="test")
    data_train, mean_train, std_train = normalize(
        data_train,
        skip_cols=["time", "event", "target_binary"])
    data_test, *_ = normalize(
        data_test, mean=mean_train, std=std_train,
        skip_cols=["time", "event", "target_binary"],
        nan_fill=0)
    data_test = data_test[mean_train.index]
    time_bins = make_time_bins(data_train["time"], event=data_train["event"])

    if not args.combined:
        data_train.drop("volume", axis=1, inplace=True)
        data_test.drop("volume", axis=1, inplace=True)
        submission_type = "emr"
    else:
        submission_type = "combined"

    if args.hparams_path:
        with open(args.hparams_path, "r") as f:
            hparams = json.load(f)
        model = train_mtlr(
            data_train.drop("target_binary", axis=1), time_bins,
            num_epochs=args.num_epochs,
            verbose=not args.quiet, **hparams)
    else:
        # find best hyperparameters and train the model
        model, best_hparams = hyperparam_search(
            data_train, time_bins, PARAM_DISTS,
            num_samples=args.hparam_samples, num_epochs=args.num_epochs,
            verbose=not args.quiet)

        with open(f"./best_hparams_{submission_type}.json", "w") as f:
            json.dump(best_hparams, f)

    # generate challenge submission
    out_path = f"./predictions_{submission_type}.csv"
    make_test_predictions(model,
                          data_test,
                          time_bins,
                          out_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("clinical_data_path", type=str, default="",
                        help="Path to CSV file with clinical data and volume.")
    parser.add_argument("--test_data_path", type=str, default="",
                        help="Path to CSV with test clinical data and volume.")
    parser.add_argument("--hparams_path", type=str, default="",
                        help="Path to JSON file with hyperparameter configuration.")
    parser.add_argument("--hparam_samples", type=int, default=60,
                        help="Number of random hyperparameter samples.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--combined", action="store_true",
                        help="Whether to train a combined model (with tumour "
                             "volume as input feature) or EMR-only (default).")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output during training.")
    args = parser.parse_args()
    main(args)
