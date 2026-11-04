import pytest

from rolling_stats.rolling_000 import rolling_mean_0


def test_rolling_mean_0_known_values():
    assert rolling_mean_0([1, 2, 3, 4]) == [1.0, 1.5, 2.0, 3.0]


def test_rolling_mean_0_zero_window_raises():
    with pytest.raises(ValueError):
        rolling_mean_0([1, 2, 3, 4], window=0)
