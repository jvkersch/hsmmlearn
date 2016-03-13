import unittest

import numpy as np
from ..emissions import GaussianEmissions, MultinomialEmissions
from ..hsmm import GaussianHSMM, MultinomialHSMM


class TestHSMMWrappers(unittest.TestCase):

    def setUp(self):
        # Exact values don't matter
        self.tmat = np.eye(3)
        self.durations = np.eye(3)

    def test_gaussian_hsmm(self):
        means = np.array([1.0, 2.0, 3.0])
        scales = np.array([0.5, 0.4, 0.3])

        hsmm = GaussianHSMM(means, scales, self.durations, self.tmat)

        self.assertIsInstance(hsmm.emissions, GaussianEmissions)
        np.testing.assert_array_equal(hsmm.emissions.means, means)
        np.testing.assert_array_equal(hsmm.emissions.scales, scales)

    def test_multinomial_hsmm(self):
        ps = np.ones((3, 5))

        hsmm = MultinomialHSMM(ps, self.durations, self.tmat)

        self.assertIsInstance(hsmm.emissions, MultinomialEmissions)
        np.testing.assert_array_equal(hsmm.emissions._probabilities, ps)
