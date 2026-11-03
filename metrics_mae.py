"""Mean absolute error metric with optional per-sample weights."""


def mean_absolute_error(y_true, y_pred, weights=None):
    """Compute the (optionally weighted) mean absolute error.

    Without weights this is mean(|y_true - y_pred|). With weights it is
    sum(w_i * |y_true_i - y_pred_i|) / sum(w_i).

    Raises ValueError if the inputs have different lengths, are empty,
    or if the weights sum to zero.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )

    if weights is None:
        weights = [1.0] * len(y_true)
    else:
        weights = list(weights)
        if len(weights) != len(y_true):
            raise ValueError(
                f"weights must have the same length as y_true "
                f"(got {len(weights)} and {len(y_true)})"
            )

    if not y_true:
        raise ValueError("cannot compute mean absolute error of empty inputs")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("sum of weights must be non-zero")

    weighted_sum = sum(
        w * abs(t - p) for t, p, w in zip(y_true, y_pred, weights)
    )
    return weighted_sum / total_weight
