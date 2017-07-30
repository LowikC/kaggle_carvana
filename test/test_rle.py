import unittest
import numpy as np
from context import carvana
from carvana import rle


class TestRLE(unittest.TestCase):
    def test_encode_one_run(self):
        mask = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        starts, lengths = rle.encode(mask)
        self.assertEquals(starts.shape[0], 1)
        self.assertEquals(starts[0], 1)
        self.assertEquals(lengths[0], 5)

if __name__ == '__main__':
    unittest.main()