"""Rolling statistics, part 78."""


def rolling_mean_0(values, window=3):
    """Rolling mean over the last `window` values (window 3)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = sum(chunk) / len(chunk)
        out.append(result)
    return out


def rolling_max_1(values, window=4):
    """Rolling max over the last `window` values (window 4)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = max(chunk)
        out.append(result)
    return out


def rolling_min_2(values, window=5):
    """Rolling min over the last `window` values (window 5)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = min(chunk)
        out.append(result)
    return out


def rolling_range_3(values, window=6):
    """Rolling range over the last `window` values (window 6)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = max(chunk) - min(chunk)
        out.append(result)
    return out


def rolling_sumsq_4(values, window=7):
    """Rolling sumsq over the last `window` values (window 7)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = sum(v * v for v in chunk)
        out.append(result)
    return out


def rolling_mean_5(values, window=8):
    """Rolling mean over the last `window` values (window 8)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = sum(chunk) / len(chunk)
        out.append(result)
    return out


def rolling_max_6(values, window=9):
    """Rolling max over the last `window` values (window 9)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = max(chunk)
        out.append(result)
    return out


def rolling_min_7(values, window=3):
    """Rolling min over the last `window` values (window 3)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = min(chunk)
        out.append(result)
    return out


def rolling_range_8(values, window=4):
    """Rolling range over the last `window` values (window 4)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = max(chunk) - min(chunk)
        out.append(result)
    return out


def rolling_sumsq_9(values, window=5):
    """Rolling sumsq over the last `window` values (window 5)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result = sum(v * v for v in chunk)
        out.append(result)
    return out


__all__ = ['rolling_mean_0', 'rolling_max_1', 'rolling_min_2', 'rolling_range_3', 'rolling_sumsq_4', 'rolling_mean_5', 'rolling_max_6', 'rolling_min_7', 'rolling_range_8', 'rolling_sumsq_9']
