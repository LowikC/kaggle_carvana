import unittest
import numpy as np
from context import carvana
from carvana import optimizers


class TestRLE(unittest.TestCase):
    def test_sgd(self):
        x, y = get_test_data()
        model = get_test_model()
        optimizer = optimizers.SGDWithAcc(lr=0.1, accum_iters=1)
        optimizer = optimizers.SGDWithAcc(lr=0.1, accum_iters=2)
        loss = lambda y_true, y_pred: y_true - y_pred
        model.compile(optimizer=)



if __name__ == '__main__':
    unittest.main()