import unittest

import numpy as np

from utils import normalize


class TestNormalize(unittest.TestCase):
    def test_scales_0_255_to_0_1(self):
        images = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        result = normalize(images)
        np.testing.assert_allclose(result, [[0.0, 1.0], [128 / 255, 64 / 255]], rtol=1e-6)

    def test_output_is_float32_in_unit_range(self):
        images = np.random.randint(0, 256, size=(5, 28, 28, 1), dtype=np.uint8)
        result = normalize(images)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, images.shape)
        self.assertTrue(result.min() >= 0.0 and result.max() <= 1.0)

    def test_does_not_modify_input(self):
        images = np.array([0, 255], dtype=np.uint8)
        normalize(images)
        np.testing.assert_array_equal(images, [0, 255])


if __name__ == "__main__":
    unittest.main()
