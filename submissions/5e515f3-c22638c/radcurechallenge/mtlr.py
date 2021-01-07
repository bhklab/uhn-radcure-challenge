from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.interpolate import interp1d

from .utils import encode_survival, make_optimizer, reset_parameters


class MTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time interval.

    Note that a slightly more efficient reformulation is used here, first proposed
    in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival distributions
    as a sequence of dependent regressors’, in Advances in neural information processing systems 24,
    2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn Consumer-Specific Reservation Price Distributions’,
    Master's thesis, University of Alberta, Edmonton, AB, 2015.
    """
    def __init__(self, in_features, num_time_bins):
        """Initialises the module.

        Parameters
        ----------
        in_features : int
            Number of input features.
        num_time_bins : int
            The number of bins to divide the time axis into.
        """
        super().__init__()
        self.in_features = in_features
        self.num_time_bins = num_time_bins

        weight = torch.zeros(self.in_features,
                             self.num_time_bins-1,
                             dtype=torch.float)
        bias = torch.zeros(self.num_time_bins-1)
        self.mtlr_weight = nn.Parameter(weight)
        self.mtlr_bias = nn.Parameter(bias)

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer("G",
                             torch.tril(torch.ones(self.num_time_bins-1,
                                                   self.num_time_bins, requires_grad=True)))
        self.reset_parameters()

    def forward(self, x):
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)


def masked_logsumexp(x, mask, dim=-1):
    """Computes logsumexp over elements of a tensor specified by a mask in a numerically stable way.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    mask : torch.Tensor
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim : int
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask, dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits, target, average=False):
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one instance
        in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used during training.
    """
    censored = target.sum(dim=1) > 1
    if censored.any():
        nll_censored = masked_logsumexp(logits[censored], target[censored]).sum()
    else:
        nll_censored = 0.
    if (~censored).any():
        nll_uncensored = (logits[~censored] * target[~censored]).sum()
    else:
        nll_uncensored = 0.

    # the normalizing constant
    norm = torch.logsumexp(logits, dim=1).sum()
    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)
    return nll_total


def mtlr_survival(logits):
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one instance
        in each row.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used during training.
    """
    # TODO: do not reallocate G in every call
    G = torch.tril(torch.ones(logits.size(1), logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)


def mtlr_survival_at_times(logits, train_times, pred_times):
    """Generates predicted survival curves at arbitrary timepoints using linear interpolation.

    This function uses scipy.interpolate internally and returns a Numpy array, in contrast
    with `mtlr_survival`.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one instance
        in each row.
    train_times : Tensor or ndarray
        Time bins used for model training. Must have the same length as the first dimension
        of `pred`.
    pred_times : np.ndarray
        Array of times used to compute the survival curve.

    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values are linearly interpolated
        at timepoints not used for training.
    """
    surv = mtlr_survival(logits).detach().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(pred_times)


def mtlr_hazard(logits):
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t, respectively.

    Parameters
    ----------
    logits : torch.Tensor
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits):
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits : torch.Tensor
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)


# training functions
def make_model(num_features, num_time_bins, hidden_sizes, dropout):
    """Returns an MTLR model with given specifications."""
    layer_sizes = [num_features] + hidden_sizes
    layers = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ELU())
        layers.append(nn.Dropout(dropout))
    layers.append(MTLR(layer_sizes[-1], num_time_bins))
    model = nn.Sequential(*layers)
    return model


def train_mtlr(data_train, time_bins,
               num_epochs=1000, lr=.01, weight_decay=0.,
               l2_reg=1., batch_size=512, hidden_sizes=[],
               dropout=0., verbose=True, device="cpu"):
    """Trains the MTLR model using minibatch gradient descent.

    Parameters
    ----------
    data_train : pd.DataFrame
        The training dataset. Must contain a `time` column with the
        event time for each sample and an `event` column containing
        the event indicator.
    num_epochs : int
        Number of training epochs.
    lr : float
        The learning rate.
    weight_decay : float
        Weight decay strength for all parameters *except* the MTLR
        weights. Only used for Deep MTLR training.
    l2_reg : float
        L2 regularization (weight decay) strenght for MTLR parameters.
    batch_size : int
        The batch size.
    hidden_sizes : list of int
        The number of hidden units in each hidden layer.
    dropout : float
        The dropout probability. Ignored if hidden_sizes is empty.
    verbose : bool
        Whether to display training progress.
    device : str
        Device name or ID to use for training.

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    x = torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(data_train["time"], data_train["event"], time_bins)
    model = make_model(x.shape[1], len(time_bins) + 1, hidden_sizes, dropout)
    optimizer = make_optimizer(model, lr, weight_decay, l2_reg)
    reset_parameters(model)
    model = model.to(device)
    model.train()

    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    for i in range(num_epochs):
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            y_pred = model(xi)
            loss = mtlr_neg_log_likelihood(y_pred, yi, average=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"[epoch {i+1: 4}/{num_epochs}] loss = {loss.item()}", end="\r")
    model.eval()
    return model
