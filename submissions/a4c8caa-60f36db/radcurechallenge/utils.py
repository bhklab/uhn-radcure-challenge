import csv
import os
from collections import deque
from math import sqrt, ceil

import torch
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pytorch_lightning.callbacks import Callback

from .model.mtlr import mtlr_density, mtlr_survival


class LogBestMetricsToCSV(Callback):
    """Logs current best epoch metrics to a CSV file."""
    def __init__(self, metric_name, save_path, maximize=True):
        self.metric_name = metric_name
        self.save_path = save_path
        self.maximize = maximize
        self.best_metric = -float("inf") if self.maximize else float("inf")
        self.metrics = {}
        self._has_logged = False

    def _is_better(self, value):
        if self.maximize:
            return value > self.best_metric
        else:
            return value < self.best_metric

    def log_metrics(self, metrics):
        with open(os.path.join(self.save_path, "best_metrics.csv"), "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    def on_epoch_end(self, trainer, pl_module):
        metrics_dict = {
            "experiment_name": trainer.logger.name,
            "version": trainer.logger.version,
            "epoch": trainer.global_step,
        }
        for k, v in trainer.callback_metrics.items():
            if "validation" in k:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                metrics_dict[k] = v
        metric_value = metrics_dict[self.metric_name]
        if not self._has_logged or self._is_better(metric_value):
            self.best_metric = metric_value
            self.log_metrics(metrics_dict)
            self._has_logged = True


class LogEarlyStopMetricsToCSV(Callback):
    def __init__(self, save_path, early_stop_patience):
        self.save_path = save_path
        self.early_stop_patience = early_stop_patience
        self._metrics = deque(maxlen=self.early_stop_patience)

    def log_metrics(self, metrics):
        with open(os.path.join(self.save_path, "best_metrics.csv"), "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    def on_epoch_end(self, trainer, pl_module):
        metrics_dict = {
            "experiment_name": trainer.logger.name,
            "version": trainer.logger.version,
            "epoch": trainer.global_step,
        }
        for k, v in trainer.callback_metrics.items():
            if "validation" in k:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                metrics_dict[k] = v
        self._metrics.append(metrics_dict)

    def on_train_end(self, trainer, pl_module):
        final_metrics = self._metrics.popleft()
        self.log_metrics(final_metrics)


def encode_survival(time, event, bins):
    time = np.clip(time, 0, bins.max())
    bin_idx = np.digitize(time, bins)
    y = torch.zeros(bins.shape[0] + 1, dtype=torch.float)
    if event:
        y[bin_idx] = 1
    else:
        y[bin_idx:] = 1
    return y


def make_time_bins(times, num_bins=None, max_time=None, use_quantiles=True):
    if num_bins is None:
        num_bins = ceil(sqrt(len(times)))
    if use_quantiles:
        bins = np.quantile(times, np.linspace(0, 1, num_bins))
    else:
        if max_time is None:
            max_time = times.max()
        bins = np.linspace(times.min(), max_time, num_bins)
    return bins


def integrated_brier_score(time_true: np.ndarray,
                           time_pred: np.ndarray,
                           event_observed: np.ndarray,
                           time_bins: np.ndarray) -> float:
    r"""Compute the integrated Brier score for a predicted survival function.

    The integrated Brier score is defined as the mean squared error between
    the true and predicted survival functions at time t, integrated over all
    timepoints.

    Parameters
    ----------
    time_true : np.ndarray, shape=(n_samples,)
        The true time to event or censoring for each sample.
    time_pred : np.ndarray, shape=(n_samples, n_time_bins)
        The predicted survival probabilities for each sample in each time bin.
    event_observed : np.ndarray, shape=(n_samples,)
        The event indicator for each sample (1 = event, 0 = censoring).
    time_bins : np.ndarray, shape=(n_time_bins,)
        The time bins for which the survival function was computed.

    Returns
    -------
    float
        The integrated Brier score of the predictions.

    Notes
    -----
    This function uses the definition from [1]_ with inverse probability
    of censoring weighting (IPCW) to correct for censored observations. The weights
    are computed using the Kaplan-Meier estimate of the censoring distribution.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher, ‘Assessment
       and comparison of prognostic classification schemes for survival data’,
       Statistics in Medicine, vol. 18, no. 17‐18, pp. 2529–2545, Sep. 1999.
    """

    # compute weights for inverse probability of censoring weighting (IPCW)
    censoring_km = KaplanMeierFitter()
    censoring_km.fit(time_true, 1 - event_observed)
    weights_event = censoring_km.survival_function_at_times(time_true).values.reshape(-1, 1)
    weights_no_event = censoring_km.survival_function_at_times(time_bins).values.reshape(1, -1)

    # scores for subjects with event before time t for each time bin
    had_event = (time_true[:, np.newaxis] <= time_bins) & event_observed[:, np.newaxis]
    scores_event = np.where(had_event, (0 - time_pred)**2 / weights_event, 0)
    # scores for subjects with no event and no censoring before time t for each time bin
    scores_no_event = np.where((time_true[:, np.newaxis] > time_bins), (1 - time_pred)**2 / weights_no_event, 0)

    scores = np.mean(scores_event + scores_no_event, axis=0)

    # integrate over all time bins
    score = np.trapz(scores, time_bins) / time_bins.max()
    return score

def plot_predictions(y_pred, time_true, event_true, time_bins):
    pred_density = mtlr_density(y_pred).detach().numpy()
    pred_survival = mtlr_survival(y_pred).detach().numpy()
    fig, ax = plt.subplots(y_pred.size(0), 2, figsize=(10, 10), sharex=True, sharey=True)
    for a, d, s, t, e in zip(ax, pred_density, pred_survival, time_true, event_true):
        a[0].bar(time_bins, d)
        a[0].axvline(np.clip(t, 0, time_bins.max()), linewidth=2, c="r" if e else "g")
        a[0].set_ylim(0, 1)
        a[0].set_xlim(0, time_bins.max() + .5)
        a[1].plot(time_bins, s)
        a[1].axvline(np.clip(t, 0, time_bins.max()), linewidth=2, c="r" if e else "g")
        a[1].set_ylim(0, 1)
        a[1].set_xlim(0, time_bins.max() + .5)
    fig.tight_layout()
    return fig


def plot_weights(weights, times, n=5):
    top_idxs = torch.argsort(weights.abs().sum(1), descending=True)[:n]
    fig, ax = plt.subplots()
    for i in top_idxs:
        ax.plot(np.pad(times, (1, 0))[:-1], weights[i])
    return fig
