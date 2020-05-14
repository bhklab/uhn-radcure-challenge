"""Metrics used in evaluation of binary and survival tasks."""
from typing import Tuple, Callable, Dict, Optional, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from joblib import Parallel, delayed

from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve,
                             precision_recall_curve, auc)

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


def permutation_test(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     metric: Callable[[np.ndarray, np.ndarray], float],
                     n_permutations: int = 5000,
                     n_jobs: int = -1,
                     **kwargs) -> Tuple[float, float]:
    r"""Compute significance of predictions using a randomized permutation test.

    The p value is computed as
    ``1/(N+1) * (sum(s(y, y_pred) >= s(perm(y), perm(y_pred)) for _ in range(N)) + 1)``
    where `s` is the performance metric and `perm` denotes a random permutation. In words,
    it is the estimated probability that a random prediction would give score at least
    as good as the actual prediction.

    Parameters
    ----------
    y_true : np.ndarray, shape=(n_samples,)
        The ground truth values.
    y_pred : np.ndarray, shape=(n_samples,)
        The model predictions.
    metric
        The performance metric.
    n_permutations, optional
        How many random permutations to use. Larger values give more
        accurate estimates but take longer to run.
    n_jobs, optional
        Number of parallel processes to use.
    **kwargs
        Additional keyword arguments passed to metric.

    Returns
    -------
    tuple of 2 floats
        The value of the performance metric and the estimated p value.
    """
    def inner():
        return metric(np.random.permutation(y_true), np.random.permutation(y_pred), **kwargs)
    estimate = metric(y_true, y_pred, **kwargs)
    perm_estimates = Parallel(n_jobs=n_jobs)(delayed(inner)() for _ in range(n_permutations))
    perm_estimates = np.array(perm_estimates)
    pval = ((perm_estimates >= estimate).sum() + 1) / (n_permutations + 1)
    return estimate, pval


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


def evaluate_binary(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    threshold: float = .5,
                    n_permutations : int = 5000,
                    n_jobs : int = -1) -> Dict[str, Union[float, List]]:
    """Compute performance metrics for a set of binary predictions.

    This function computes the confusion matrix, area under the ROC curve
    (AUC) and average precision (AP, a variant of area under the
    precision-recall curve). Significance of AUC and AP is computed using
    a permutation test.

    Parameters
    ----------
    y_true : np.ndarray, shape=(n_samples,)
        The true binary class labels (1=positive class).
    y_pred : np.ndarray, shape=(n_samples,)
        The predicted class probablities/scores.
    threshold, optional
        The classification threshold for model outputs. Only used
        for computing the confusion matrix.
    n_permutations, optional
        How many random permutations to use. Larger values give more
        accurate estimates but take longer to run.
    n_jobs, optional
        Number of parallel processes to use.

    Returns
    -------
    dict
        The computed performance metrics.
    """

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    conf_matrix = confusion_matrix(y_true, y_pred > threshold)
    auc, auc_pval = permutation_test(y_true, y_pred,
                                     roc_auc_score, n_permutations, n_jobs)
    avg_prec, avg_prec_pval = permutation_test(y_true, y_pred,
                                               average_precision_score,
                                               n_permutations, n_jobs)
    return {
        "confusion_matrix": conf_matrix,
        "roc_auc": auc,
        "roc_auc_pval": auc_pval,
        "average_precision": avg_prec,
        "average_precision_pval": avg_prec_pval
    }


def evaluate_survival(event_true: np.ndarray,
                      time_true: np.ndarray,
                      event_pred: np.ndarray,
                      time_pred: Optional[np.ndarray] = None,
                      n_permutations: int = 5000,
                      n_jobs: int = -1) -> Dict[str, float]:
    """Compute performance metrics for a set of survival predictions.

    This function computes the concordance index for event risk predicitons
    and, if `time_pred` is passed, the integrated Brier score for the predicted
    survival function. Significance of both metrics is computed using
    a permutation test.

    Parameters
    ----------
    event_true : np.ndarray, shape=(n_samples,)
        The event indicator for each sample (1 = event, 0 = censoring).
    time_true : np.ndarray, shape=(n_samples,)
        The true time to event or censoring for each sample.
    event_pred : np.ndarray, shape=(n_samples,)
        The predicted risk scores for each sample (greater = higher risk).
    time_pred : np.ndarray, optional, shape=(n_samples, 23)
        The predicted survival probabilities for each sample in each time bin.
        The predictions should be computed for time bins spaced 1 month apart
        up to 2 years.
    n_permutations, optional
        How many random permutations to use. Larger values give more
        accurate estimates but take longer to run.
    n_jobs, optional
        Number of parallel processes to use.

    Notes
    -----
    This function assumes the survival function predictions are made at
    23 equally-spaced time bins from 1 month to 1 year. Therefore, each
    for each `i`, ``time_pred[i]`` should be the predicted survival
    probability at time ``time_bins[i-1] < t <= time_bins[i]``.

    Returns
    -------
    dict
        The computed performance metrics.
    """

    # evaluate predictions at time < 2 years
    # to account for the shorter follow-up in the
    # test set
    event_observed = event_true.copy()
    event_observed[time_true > 2] = 0
    time_observed = np.clip(time_true, 0, 2)
    ci, ci_pval = permutation_test(time_observed, -event_pred,
                                   concordance_index,
                                   n_permutations, n_jobs,
                                   event_observed=event_observed)
    if time_pred is not None:
        time_bins = np.linspace(1, 2, 23)
        if time_pred.shape[1] != len(time_bins):
            raise ValueError((f"Expected predictions at {len(time_bins)}"
                              f" timepoints, got {time_pred.shape[1]}."))
        brier, brier_pval = permutation_test(time_true, time_pred,
                                             integrated_brier_score,
                                             event_observed=event_true,
                                             time_bins=time_bins)

    return {
        "concordance_index": ci,
        "concordance_index_pval": ci_pval,
        "integrated_brier_score": brier,
        "integrated_brier_score_pval": brier_pval
    }


def plot_roc_curve(true: Union[np.ndarray, pd.Series],
                   predicted: Union[np.ndarray, pd.Series],
                   label: str = None,
                   ax: Optional[Axes] = None) -> Axes:
    """Plot the receiver operating characteristic (ROC) curve for a set of
    binary predictions.

    Parameters
    ----------
    true
        The ground truth binary labels.
    predicted
        The predicted positive class scores.
    label
        The label used in plot legend.
    ax
        Axes object to plot on. If None, a new set of axes is created.

    Returns
    -------
    Axes
        The ROC curve plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    fpr, tpr, _ = roc_curve(true, predicted)
    auc_val = auc(fpr, tpr)
    ax.plot([0, 1], [0, 1], c="grey", linestyle="--")
    if label:
        label += f" (AUC = {auc_val:.2f})"
    else:
        label = f"AUC = {auc_val:.2f}"
    ax.plot(fpr, tpr, label=label)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    return ax


def plot_pr_curve(true: Union[np.ndarray, pd.Series],
                  predicted: Union[np.ndarray, pd.Series],
                  label: str = None,
                  ax: Optional[Axes] = None) -> Axes:
    """Plot the precision-recall (PR) curve for a set of binary predictions.

    Parameters
    ----------
    true
        The ground truth binary labels.
    predicted
        The predicted positive class scores.
    label
        The label used in plot legend.
    ax
        Axes object to plot on. If None, a new set of axes is created.

    Returns
    -------
    Axes
        The PR curve plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    precision, recall, _ = precision_recall_curve(true, predicted)
    ap_val = average_precision_score(true, predicted)
    if label:
        label += f" (AP = {ap_val:.2f})"
    else:
        label = f"AP = {ap_val:.2f}"
    ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    return ax
