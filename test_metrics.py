import pytest

from metrics import accuracy, precision


def test_accuracy_partial_match():
    assert accuracy([1, 2, 3, 4], [1, 0, 3, 0]) == 0.5


def test_accuracy_perfect_and_none():
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert accuracy([1, 1, 1], [0, 0, 0]) == 0.0


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])


def test_precision_counts_only_predicted_positives():
    # predicted positives at indices 0, 2, 3 -> truly positive at 0, 3 -> 2/3
    assert precision([1, 0, 0, 1], [1, 0, 1, 1]) == pytest.approx(2 / 3)


def test_precision_no_predicted_positives_raises():
    with pytest.raises(ValueError):
        precision([1, 0], [0, 0])
