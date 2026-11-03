"""F1 score for binary classification."""


def f1_score(y_true, y_pred, positive=1):
    """Return the F1 score for the given positive label.

    F1 is the harmonic mean of precision and recall, computed as
    2 * TP / (2 * TP + FP + FN). If there are no true positives and no
    false positives/negatives (so the denominator is zero), 0.0 is returned.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive:
            fp += 1
        elif t == positive:
            fn += 1

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return 2 * tp / denom
