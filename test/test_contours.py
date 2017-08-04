import unittest
import numpy as np
from context import carvana
from carvana import contours


class TestRLE(unittest.TestCase):
    def test_contours(self):
        mask = np.zeros((8, 12), dtype=np.uint8)
        mask[4:7, 5:9] = 1
        expected_mask = np.zeros((8, 12), dtype=np.uint8)
        expected_mask[5:6, 6:8] = 2

        expected_mask[4:7, 5] = 3
        expected_mask[4:7, 8] = 3
        expected_mask[4, 5:9] = 3
        expected_mask[6, 5:9] = 3

        expected_mask[4:7, 4] = 1
        expected_mask[4:7, 9] = 1
        expected_mask[3, 5:9] = 1
        expected_mask[7, 5:9] = 1

        mask_contours = contours.get_contours(mask)

        self.assertTrue(np.all(mask_contours == expected_mask))


if __name__ == '__main__':
    unittest.main()