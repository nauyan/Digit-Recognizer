def accuracy(y_true, y_pred):
    """Return the fraction of positions where y_true and y_pred match.

    Raises ValueError if the lists have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true has {len(y_true)} items, "
            f"y_pred has {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot compute accuracy of empty inputs")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision(y_true, y_pred, positive=1):
    """Return the fraction of predicted positives that are truly positive.

    Computed as true positives / (true positives + false positives), where
    a position counts as a predicted positive when y_pred equals `positive`.
    Returns 0.0 when nothing is predicted as positive.

    Raises ValueError if the lists have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true has {len(y_true)} items, "
            f"y_pred has {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot compute precision of empty inputs")

    predicted_positive = [
        t for t, p in zip(y_true, y_pred) if p == positive
    ]
    if not predicted_positive:
        return 0.0

    true_positives = sum(1 for t in predicted_positive if t == positive)
    return true_positives / len(predicted_positive)
