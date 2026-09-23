"""Forecast error and calibration metrics."""

import numpy as np


def rmse(a, b) -> float:
    """Root-mean-square error between two index-aligned Series (kWh)."""
    return float(np.sqrt(((a - b) ** 2).mean()))


def mae(a, b) -> float:
    """Mean absolute error between two index-aligned Series (kWh)."""
    return float((a - b).abs().mean())


def interval_coverage(actual, lower, upper) -> float:
    """Fraction of ``actual`` values inside the closed interval [lower, upper]."""
    return float(((actual >= lower) & (actual <= upper)).mean())


def pinball_loss(actual, pred, q) -> float:
    """Mean pinball (quantile) loss of ``pred`` at level ``q``; lower is better."""
    err = actual.values - pred.values
    return float(np.where(err >= 0, q * err, (q - 1) * err).mean())
