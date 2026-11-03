import math

import numpy as np
import pytest

from metrics_rmse import rmse


def test_perfect_prediction_is_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_known_value():
    # errors: 1, 1, 1, 1 -> mean squared 1 -> rmse 1
    assert rmse([0, 0, 0, 0], [1, -1, 1, -1]) == pytest.approx(1.0)


def test_matches_manual_formula():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    expected = math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true))
    assert rmse(y_true, y_pred) == pytest.approx(expected)


def test_accepts_numpy_arrays():
    y = np.linspace(0, 1, 50)
    assert rmse(y, y + 0.1) == pytest.approx(0.1)


def test_returns_float():
    assert isinstance(rmse([1, 2], [2, 3]), float)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse([1, 2, 3], [1, 2])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        rmse([], [])
