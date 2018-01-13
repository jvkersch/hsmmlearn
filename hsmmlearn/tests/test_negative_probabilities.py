import unittest

import numpy as np
import numpy.testing as nptest

from hsmmlearn.hsmm import MultinomialHSMM


class TestNegativeProbabilities(unittest.TestCase):

    def test_negative(self):
        # Regression test for GH # 34.

        # Given
        prior = np.array([[0.1, 0.1, 0.1, 0.7],
                          [0.1, 0.1, 0.7, 0.1],
                          [0.1, 0.7, 0.1, 0.1]])
        durations = np.array([[0.1, 0.1, 0.8, 0.0, 0.0],
                              [0.1, 0.7, 0.2, 0.0, 0.0],
                              [0.2, 0.2, 0.2, 0.2, 0.2]])
        tmat = np.array([[0.0, 0.5, 0.5],
                         [0.3, 0.0, 0.7],
                         [0.6, 0.4, 0.0]])
        samples = np.array(3*(5*[0] + 5*[1] + 5*[2] + 5*[3]))
        hsmm = MultinomialHSMM(prior, durations, tmat)

        # When
        hsmm.fit(samples)

        # Then
        for prob in hsmm.emissions._probabilities.ravel():
            self.assertGreaterEqual(prob, 0)
