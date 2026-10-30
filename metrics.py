def accuracy(y_true, y_pred):
    """Return the fraction of positions where y_true and y_pred match."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")
    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return matches / len(y_true)


def precision(y_true, y_pred, positive=1):
    """Return TP / (TP + FP) for the given positive class.

    Returns 0.0 when the model predicts no positives.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if p == positive and t == positive)
    pred_pos = sum(1 for p in y_pred if p == positive)
    if pred_pos == 0:
        return 0.0
    return true_pos / pred_pos
