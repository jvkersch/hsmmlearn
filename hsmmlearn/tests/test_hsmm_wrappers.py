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

    def test_gaussian_hsmm_means_scales(self):
        means = np.array([1.0, 2.0, 3.0])
        scales = np.array([0.5, 0.4, 0.3])

        hsmm = GaussianHSMM(means, scales, self.durations, self.tmat)

        # Test property getters
        np.testing.assert_array_equal(hsmm.means, means)
        np.testing.assert_array_equal(hsmm.scales, scales)

        # Now update properties and check that the value changed on the
        # emissions.
        new_means = np.array([5.0, 5.0, 5.0])
        new_scales = np.array([1.0, 1.0, 1.0])

        hsmm.means = new_means
        hsmm.scales = new_scales

        emissions = hsmm.emissions
        np.testing.assert_array_equal(emissions.means, new_means)
        np.testing.assert_array_equal(emissions.scales, new_scales)

    def test_multinomial_hsmm(self):
        ps = np.ones((3, 5))

        hsmm = MultinomialHSMM(ps, self.durations, self.tmat)

        self.assertIsInstance(hsmm.emissions, MultinomialEmissions)
        np.testing.assert_array_equal(hsmm.emissions._probabilities, ps)
