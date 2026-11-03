import unittest

from metrics import accuracy, precision


class TestAccuracy(unittest.TestCase):
    def test_partial_match(self):
        self.assertEqual(accuracy([1, 2, 3, 4], [1, 0, 3, 0]), 0.5)

    def test_perfect_and_zero_match(self):
        self.assertEqual(accuracy([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertEqual(accuracy(["a", "b"], ["x", "y"]), 0.0)


class TestPrecision(unittest.TestCase):
    def test_partial_precision(self):
        # Predicted positives at indices 0, 2, 3; only 0 and 3 are truly positive.
        self.assertAlmostEqual(
            precision([1, 0, 0, 1], [1, 0, 1, 1]), 2 / 3
        )

    def test_custom_positive_label(self):
        self.assertEqual(
            precision(["cat", "dog", "cat"], ["cat", "cat", "dog"], positive="cat"),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
