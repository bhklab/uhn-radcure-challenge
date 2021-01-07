import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn


def masked_logsumexp(x: torch.Tensor,
                     mask: torch.Tensor,
                     dim: int = -1
                     ) -> torch.Tensor:
    """Computes the logsumexp of input elements indicated by the mask in a
    numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        Tensor with 1s at positions used to compute the result and 0s everywhere
        else.
    dim
        Dimension along which to compute the logsumexp.

    Returns
    -------
    torch.Tensor
        The logsumexp value."""
    max_val, _ = (x * mask).max(dim=dim)
    return torch.log(torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask, dim=dim)) + max_val


def mtlr_log_density(y_pred: torch.Tensor) -> torch.Tensor:
    """Computes the log density of the MTLR model given predicted logits.

    Parameters
    ----------
    y_pred
        Predicted logits as returned by `MTLRLayer.forward()`.

    Returns
    -------
    torch.Tensor
        The log density values.
    """
    G = torch.tril(torch.ones(y_pred.size(1), y_pred.size(1) + 1))
    logit_sums = torch.matmul(y_pred, G)
    return torch.log_softmax(logit_sums, dim=1)


def mtlr_density(y_pred: torch.Tensor) -> torch.Tensor:
    """Computes the density of the MTLR model given predicted logits.

    Parameters
    ----------
    y_pred
        Predicted logits as returned by `MTLRLayer.forward()`.

    Returns
    -------
    torch.Tensor
        The density values.
    """
    return mtlr_log_density(y_pred).exp()


def mtlr_survival(y_pred: torch.Tensor) -> torch.Tensor:
    G = torch.tril(torch.ones(y_pred.size(1) + 1, y_pred.size(1) + 1))
    density = mtlr_density(y_pred)
    return torch.matmul(density, G)


def mtlr_survival_at_times(y_pred, times, pred_times):
    surv = mtlr_survival(y_pred).detach().numpy()
    interpolator = interp1d(times, surv)
    pred_times = np.clip(pred_times, 0, times.max())
    return interpolator(pred_times)


def mtlr_hazard(y_pred):
    return mtlr_density(y_pred)[:, :-1] / (mtlr_survival(y_pred) + 1e-15)[:, 1:]


def mtlr_risk(y_pred, bins=None):
    hazard = mtlr_hazard(y_pred)
    if bins is None:
        bins = torch.arange(hazard.size(1))
    return torch.trapz(hazard.cumsum(1), bins, dim=1)


def mtlr_loss(y_pred, y_true):
    G = torch.tril(torch.ones(y_pred.size(1), y_pred.size(1) + 1))
    logit_sums = torch.matmul(y_pred, G)
    return -torch.sum(masked_logsumexp(logit_sums, y_true) - torch.logsumexp(logit_sums, dim=1)) / y_true.size(0)


class MTLRLayer(nn.Module):
    def __init__(self, in_features, num_time_bins):
        super().__init__()
        self.in_features = in_features
        self.num_time_bins = num_time_bins

        weight = torch.zeros(self.in_features,
                             self.num_time_bins-1,
                             dtype=torch.float)
        bias = torch.zeros(self.num_time_bins-1)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.reset_parameters()

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.constant_(self.bias, 0.)


class MTLRLoss(nn.Module):
    def __init__(self, num_time_bins, weights=None):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.weights = weights
        self.register_buffer("G", torch.tril(torch.ones(self.num_time_bins - 1, self.num_time_bins)))

    def __call__(self, y_pred, y_true):
        logit_sums = torch.matmul(y_pred, self.G)
        return -torch.mean(masked_logsumexp(logit_sums, y_true) - torch.logsumexp(logit_sums, dim=1), dim=0)
