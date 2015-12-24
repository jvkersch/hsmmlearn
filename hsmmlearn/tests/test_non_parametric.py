import six
import unittest

import numpy as np
from ..utils import NonParametricDistribution


class TestNonParametric(unittest.TestCase):

    def test_sample_rvs(self):
        dist = NonParametricDistribution([1, 10], [.5, .5])
        size = 10000
        samples = dist.rvs(size)

        self.assertEqual(samples.shape, (size, ))
        six.assertCountEqual(self, [1, 10], np.unique(samples))

    def test_corner_cases(self):
        # TODO check message
        with self.assertRaises(ValueError):
            NonParametricDistribution([1, 10], [.5])
        with self.assertRaises(ValueError):
            NonParametricDistribution([[1, 10]], [.5, .5])
        with self.assertRaises(ValueError):
            NonParametricDistribution([1, 10], [[.5, .5]])
