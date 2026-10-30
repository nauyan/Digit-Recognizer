def accuracy(y_true, y_pred):
    """Return the fraction of positions where y_true and y_pred match.

    Raises ValueError if the lists differ in length or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )
    if len(y_true) == 0:
        raise ValueError("accuracy is undefined for empty inputs")

    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return matches / len(y_true)


def precision(y_true, y_pred, positive=1):
    """Return the fraction of predicted positives that are truly positive.

    Computed as TP / (TP + FP), where a position counts as a predicted
    positive when y_pred equals `positive`.

    Raises ValueError if the lists differ in length, or if there are no
    predicted positives (precision is undefined there).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )

    predicted_positive = [t for t, p in zip(y_true, y_pred) if p == positive]
    if not predicted_positive:
        raise ValueError("precision is undefined when there are no predicted positives")

    true_positives = sum(1 for t in predicted_positive if t == positive)
    return true_positives / len(predicted_positive)
