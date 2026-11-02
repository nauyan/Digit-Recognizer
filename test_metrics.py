import pytest

from metrics import accuracy, precision


def test_accuracy_partial_match():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_accuracy_perfect_and_zero():
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert accuracy([1, 2, 3], [4, 5, 6]) == 0.0


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])


def test_precision_mixed():
    # Predicted positives at indices 0, 1, 2; only 0 and 2 are true positives.
    assert precision([1, 0, 1, 0], [1, 1, 1, 0]) == pytest.approx(2 / 3)


def test_precision_no_predicted_positives_is_zero():
    assert precision([1, 1], [0, 0]) == 0.0


def test_precision_custom_positive_label():
    assert precision(["spam", "ham", "spam"], ["spam", "spam", "ham"], positive="spam") == 0.5
