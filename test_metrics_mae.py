import pytest

from metrics_mae import mean_absolute_error


def test_unweighted_mae():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0
    assert mean_absolute_error([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_unweighted_matches_explicit_weights_of_ones():
    y_true = [3.0, -1.0, 2.5]
    y_pred = [2.0, 0.0, 4.0]
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(
        mean_absolute_error(y_true, y_pred, weights=[1, 1, 1])
    )


def test_weighted_mae():
    # errors: 1, 3 ; weights: 1, 3 -> (1*1 + 3*3) / 4 = 2.5
    assert mean_absolute_error([0, 0], [1, 3], weights=[1, 3]) == pytest.approx(2.5)


def test_zero_weight_ignores_sample():
    assert mean_absolute_error([0, 0], [100, 2], weights=[0, 1]) == pytest.approx(2.0)


def test_scaling_weights_does_not_change_result():
    y_true = [1, 2, 3]
    y_pred = [2, 4, 1]
    base = mean_absolute_error(y_true, y_pred, weights=[1, 2, 3])
    scaled = mean_absolute_error(y_true, y_pred, weights=[10, 20, 30])
    assert scaled == pytest.approx(base)


def test_accepts_tuples_and_generators():
    assert mean_absolute_error((1, 2), (2, 4)) == pytest.approx(1.5)
    assert mean_absolute_error(iter([1, 2]), iter([2, 4])) == pytest.approx(1.5)


def test_mismatched_y_lengths_raise():
    with pytest.raises(ValueError):
        mean_absolute_error([1, 2, 3], [1, 2])


def test_mismatched_weights_length_raises():
    with pytest.raises(ValueError):
        mean_absolute_error([1, 2], [1, 2], weights=[1])


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_all_zero_weights_raise():
    with pytest.raises(ValueError):
        mean_absolute_error([1, 2], [1, 2], weights=[0, 0])
