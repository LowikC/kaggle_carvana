import unittest
import numpy as np
from context import carvana
from carvana import contours


class TestRLE(unittest.TestCase):
    @staticmethod
    def get_test_mask():
        mask = np.zeros((8, 12), dtype=np.uint8)
        mask[4:7, 5:9] = 1
        return mask

    @staticmethod
    def get_expected_contours():
        expected_contours = np.zeros((8, 12), dtype=np.uint8)
        expected_contours[5:6, 6:8] = 2

        expected_contours[4:7, 5] = 3
        expected_contours[4:7, 8] = 3
        expected_contours[4, 5:9] = 3
        expected_contours[6, 5:9] = 3

        expected_contours[4:7, 4] = 1
        expected_contours[4:7, 9] = 1
        expected_contours[3, 5:9] = 1
        expected_contours[7, 5:9] = 1
        return expected_contours

    @staticmethod
    def get_expected_weights(weights):
        expected_weights = np.ones((8, 12), dtype=np.float32) * weights[0]
        expected_weights[5:6, 6:8] = weights[2]

        expected_weights[4:7, 5] = weights[3]
        expected_weights[4:7, 8] = weights[3]
        expected_weights[4, 5:9] = weights[3]
        expected_weights[6, 5:9] = weights[3]

        expected_weights[4:7, 4] = weights[1]
        expected_weights[4:7, 9] = weights[1]
        expected_weights[3, 5:9] = weights[1]
        expected_weights[7, 5:9] = weights[1]
        return expected_weights

    def test_contours(self):
        # Setup
        mask = self.get_test_mask()
        expected_contours = self.get_expected_contours()

        # Action
        mask_contours = contours.get_contours(mask)

        # Check
        self.assertTrue(np.all(mask_contours == expected_contours))

    def test_contours_weights(self):
        # Setup
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        mask = self.get_test_mask()
        expected_weights = self.get_expected_weights(weights)

        # Action
        mask_weights = contours.get_pixel_weights(mask, weights, thickness=1)

        # Check
        self.assertTrue(np.all(mask_weights == expected_weights))

    def test_contours_batch(self):
        # Setup
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        mask = self.get_test_mask()
        expected_weights = self.get_expected_weights(weights)

        y_batch = np.zeros((2, 8, 12, 1), dtype=np.uint8)
        y_batch[0, :, :, 0] = mask
        y_batch[1, :, :, 0] = mask

        expected_y = np.zeros((2, 8, 12, 2))
        expected_y[0, :, :, 0] = mask
        expected_y[0, :, :, 1] = expected_weights
        expected_y[1, :, :, 0] = mask
        expected_y[1, :, :, 1] = expected_weights

        # Action
        y_prep = contours.get_weighted_mask_batch(y_batch, weights, thickness=1)

        # Check
        self.assertTrue(np.all(y_prep == expected_y))

if __name__ == '__main__':
    unittest.main()
