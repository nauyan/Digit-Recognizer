def mean_absolute_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
