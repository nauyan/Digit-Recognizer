from metrics import accuracy, precision


def test_accuracy_all_correct():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0


def test_accuracy_partial_match():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_precision_mixed():
    # predicted positives at indices 0, 1, 2; true positives at 0 and 2
    assert precision([1, 0, 1, 0], [1, 1, 1, 0]) == 2 / 3
