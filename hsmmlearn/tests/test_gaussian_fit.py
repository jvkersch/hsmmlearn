import unittest

import numpy as np

from ..hsmm import HSMMModel
from ..emissions import GaussianEmissions


class TestGaussianFit(unittest.TestCase):

    def test_corner_case_all_constant(self):
        np.random.seed(1234)

        means = np.array([0.0, 5.0, 10.0])
        scales = np.ones_like(means)
        durations = np.array([
            [0.1, 0.0, 0.0, 0.9],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.0, 0.0, 0.9]
        ])
        tmat = np.array([
            [0.2, 0.4, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.4, 0.1]
        ])
        observations = np.random.randn(12)
        observations[4:8] += 5.0

        hsmm = HSMMModel(
            GaussianEmissions(means, scales), durations, tmat,
        )
        has_converged, log_likelihood = hsmm.fit(observations)

        self.assertTrue(has_converged)
        np.testing.assert_almost_equal(log_likelihood, -2.22642, decimal=4)

        np.testing.assert_almost_equal(
            hsmm.tmat, np.array([[0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [1.0, 0.0, 0.0]])
        )

        np.testing.assert_almost_equal(
            hsmm.durations,
            np.array([[0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0]])
        )
        np.testing.assert_array_almost_equal(
            hsmm.emissions.means,
            np.array([0.03943846, 5.34205421, 4.3634765])
        )
        np.testing.assert_array_almost_equal(
            hsmm.emissions.scales,
            np.array([1.17872032e+00, 7.51486350e-01, 3.89204133e-07])
        )
