"""Root mean squared error metric."""

import numpy as np


def rmse(y_true, y_pred):
    """Return the root mean squared error between two equal-length sequences.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        float: sqrt(mean((y_true - y_pred) ** 2)).

    Raises:
        ValueError: If the inputs have different shapes or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("rmse requires at least one value")
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))
