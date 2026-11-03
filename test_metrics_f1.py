import pytest

from metrics_f1 import f1_score


def test_perfect_predictions():
    assert f1_score([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_all_wrong():
    assert f1_score([1, 1, 0], [0, 0, 1]) == pytest.approx(0.0)


def test_mixed_case():
    # TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3, F1=2/3
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_matches_precision_recall_formula():
    # TP=3, FP=1, FN=2 -> P=3/4, R=3/5, F1=2PR/(P+R)=2/3
    y_true = [1, 1, 1, 1, 1, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 0]
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_no_positives_anywhere_returns_zero():
    assert f1_score([0, 0, 0], [0, 0, 0]) == 0.0


def test_custom_positive_label():
    y_true = ["cat", "dog", "cat", "dog"]
    y_pred = ["cat", "cat", "cat", "dog"]
    # TP=2, FP=1, FN=0 -> F1 = 4/5
    assert f1_score(y_true, y_pred, positive="cat") == pytest.approx(4 / 5)


def test_default_positive_is_one():
    y_true = [0, 1, 1]
    y_pred = [0, 1, 0]
    assert f1_score(y_true, y_pred) == f1_score(y_true, y_pred, positive=1)


def test_accepts_numpy_like_iterables():
    y_true = (1, 0, 1)
    y_pred = iter([1, 1, 1])
    # TP=2, FP=1, FN=0 -> F1 = 4/5
    assert f1_score(y_true, y_pred) == pytest.approx(4 / 5)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        f1_score([1, 0], [1])
