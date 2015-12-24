import csv
import os
import unittest

from scipy.stats import norm
import numpy as np
from simple_hsmm.base import _viterbi_impl


def load_data(fname):
    fname = os.path.join(os.path.dirname(__file__), fname)
    with open(fname) as fp:
        reader = csv.reader(fp)
        next(reader)
        data = [float(row[0]) for row in reader]
    return np.array(data)


class TestBase(unittest.TestCase):

    def test_viterbi_impl(self):

        # Viterbi parameters (transcribed from test_base.R).
        mean = np.array([-1.5, 0, 1.5])
        variance = np.array([0.5, 0.6, 0.8])

        rd = np.array([[0.1, 0.7, 0.2],
                       [0.2, 0.6, 0.2],
                       [0.3, 0.5, 0.2]])
        tpm = np.array([[0, 0.5, 0.5],
                        [0.7, 0, 0.3],
                        [0.8, 0.2, 0]])
        pi = np.full(3, 1.0 / 3)

        # Compute log likelihoods of observations given the states.
        observations = load_data('viterbi_observations.csv')
        ll_obs = norm.pdf(
            observations, loc=mean[:, np.newaxis],
            scale=variance[:, np.newaxis] ** 0.5
        )

        tau = observations.shape[0]
        j, m = rd.shape

        # Output array.
        states = np.empty(tau, dtype=np.int32)

        # When
        _viterbi_impl(tau, j, m, rd.ravel(),
                      tpm.ravel(), pi, ll_obs.ravel(), states)

        # Then
        expected_states = load_data('viterbi_states.csv')
        expected_states = (expected_states - 1).astype(int)  # 1-based.
        np.testing.assert_array_equal(states, expected_states)
