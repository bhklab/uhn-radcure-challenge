from math import sqrt, ceil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam


def make_data(path, split="training"):
    """Load and preprocess the data."""
    data = (pd.read_csv(path, index_col="Study ID")
            .query("split == @split")
            .drop(["cancer_death", "split"], axis=1, errors="ignore"))
    data = data.rename(columns={"death": "event", "survival_time": "time"})
    # Convert time to months
    data["time"] *= 12
    # binarize T stage as T1/2 = 0, T3/4 = 1
    data["T Stage"] = data["T Stage"].map(
        lambda x: "T1/2" if x in ["T1", "T1a", "T1b", "T2"] else "T3/4",
        na_action="ignore")
    # use more fine-grained grouping for N stage
    data["N Stage"] = data["N Stage"].str.slice(0, 2)
    data["Stage"] = data["Stage"].map(
        lambda x: "I/II" if x in ["I", "II", "IIA"] else "III/IV",
        na_action="ignore")
    data["ECOG"] = data["ECOG"].map(
        lambda x: ">0" if x > 0 else "0", na_action="ignore")
    data = pd.get_dummies(data,
                          columns=[
                              "Sex",
                              "N Stage",
                              "Disease Site"
                          ],
                          drop_first=True)
    # keep all indicator columns in case of missing values
    data = pd.get_dummies(data,
                          columns=[
                              "HPV Combined",
                              "T Stage",
                              "Stage",
                              "ECOG"
                          ])
    return data


def encode_survival(time, event, bins):
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time : np.ndarray
        Array of event or censoring times.
    event : np.ndarray
        Array of event indicators (0 = censored).
    bins : np.ndarray
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    time = np.clip(time, 0, bins.max())
    bin_idxs = np.digitize(time, bins)
    # add extra bin [max_time, inf) at the end
    y = np.zeros((time.shape[0], bins.shape[0] + 1), dtype=np.int)
    for i, e in enumerate(event):
        bin_idx = bin_idxs[i]
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return torch.tensor(y, dtype=torch.float)


def reset_parameters(model):
    """Resets the parameters of a PyTorch module and its children."""
    for m in model.modules():
        try:
            m.reset_parameters()
        except AttributeError:
            continue
    return model


def make_optimizer(model, lr, weight_decay, l2_reg):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = Adam([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": l2_reg},
    ], lr=lr, weight_decay=weight_decay)
    return optimizer


def make_time_bins(times, num_bins=None, use_quantiles=True, event=None):
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times : np.ndarray
        Array of survival times.
    num_bins : int, optional
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles : bool
        If True, the bin edges will correspond to quantiles of `times` (default).
        Otherwise, generates equally-spaced bins.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    if event is not None:
        times = times[event > 0]
    if num_bins is None:
        num_bins = ceil(sqrt(len(times)))
    if use_quantiles:
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    return bins


def normalize(data, mean=None, std=None, skip_cols=[], nan_fill=None):
    """Normalizes the columns of Pandas DataFrame to zero mean and unit
    standard deviation."""
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    if skip_cols is not None:
        mean[skip_cols] = 0
        std[skip_cols] = 1
    data_norm = (data - mean) / std
    if nan_fill is not None:
        data_norm = data_norm.fillna(nan_fill)
    return data_norm, mean, std
