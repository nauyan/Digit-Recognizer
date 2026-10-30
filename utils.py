import numpy as np


def normalize(images):
    """Scale pixel values from the 0-255 range to the 0-1 range.

    Returns a new float32 array; the input is not modified.
    """
    return np.asarray(images, dtype=np.float32) / 255.0
