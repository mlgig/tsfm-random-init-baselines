"""Tests for deterministic handling of raw UEA ragged collections."""

from __future__ import annotations

import unittest

import numpy as np

from common.uea import pad_train_test_to_common_length, right_pad_collection


class UeaCollectionTest(unittest.TestCase):
    def test_right_padding_preserves_observed_values_and_order(self) -> None:
        collection = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        ]

        padded = right_pad_collection(collection)

        self.assertEqual(padded.shape, (2, 2, 3))
        np.testing.assert_array_equal(padded[0], [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
        np.testing.assert_array_equal(padded[1], collection[1])
        self.assertEqual(padded.dtype, np.float32)

    def test_train_and_test_use_one_common_length(self) -> None:
        train = [np.ones((2, 2)), np.full((2, 3), 2.0)]
        test = [np.full((2, 4), 3.0)]

        padded_train, padded_test = pad_train_test_to_common_length(train, test)

        self.assertEqual(padded_train.shape, (2, 2, 4))
        self.assertEqual(padded_test.shape, (1, 2, 4))
        np.testing.assert_array_equal(padded_train[0, :, 2:], 0.0)
        np.testing.assert_array_equal(padded_train[1, :, :3], 2.0)


if __name__ == "__main__":
    unittest.main()
