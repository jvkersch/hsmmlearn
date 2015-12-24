import unittest

import numpy as np

from ..hsmm import HSMMModel
from ..emissions import MultinomialEmissions


class TestMultivariateFit(unittest.TestCase):

    def test_corner_case_all_ones(self):
        emissions = np.array([
            [0.1, 0.9, 0.0, 0.0],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.0, 0.0, 0.9]
        ])
        durations = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.9, 0.0, 0.0]
        ])
        tmat = np.array([
            [0.2, 0.4, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.4, 0.1]
        ])
        observations = np.ones(10, dtype=int)
        hsmm = HSMMModel(
            MultinomialEmissions(emissions), durations, tmat,
        )

        has_converged, log_likelihood = hsmm.fit(observations)

        self.assertTrue(has_converged)
        self.assertLess(abs(log_likelihood), 1e-15)

        np.testing.assert_almost_equal(
            hsmm.tmat, np.array([[1.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0]])
        )

        np.testing.assert_almost_equal(
            hsmm.emissions._probabilities,
            np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        )

    def test_corner_case_transitions(self):
        emissions = np.eye(3)
        durations = np.zeros((3, 2))
        durations[:, 0] = 0.1
        durations[:, 1] = 0.9
        tmat = np.ones((3, 3))
        tmat /= tmat.sum(axis=1)

        hsmm = HSMMModel(
            MultinomialEmissions(emissions), durations, tmat,
        )
        obs = np.array(10 * [0, 0, 1, 1, 2, 2])

        has_converged, log_likelihood = hsmm.fit(obs)

        self.assertTrue(has_converged)
        self.assertLess(abs(log_likelihood), 1e-9)

        np.testing.assert_array_almost_equal(
            hsmm.tmat,
            np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0]])
        )
        np.testing.assert_array_almost_equal(
            hsmm.durations,
            np.array([[0.0, 1.0],
                      [0.0, 1.0],
                      [0.0, 1.0]])
        )
        np.testing.assert_array_almost_equal(
            hsmm.emissions._probabilities,
            np.eye(3)
        )
        np.testing.assert_array_almost_equal(
            hsmm._startprob,
            np.array([1.0, 0.0, 0.0])
        )
