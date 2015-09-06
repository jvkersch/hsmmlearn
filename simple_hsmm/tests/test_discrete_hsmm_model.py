from collections import Counter
from itertools import groupby
import unittest

import numpy as np

from simple_hsmm.hsmm import DiscreteHSMMModel


class TestDiscreteHSMMModelSampling(unittest.TestCase):

    def test_sample_single(self):
        emissions = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        durations = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.9, 0.0, 0.0]
        ])
        tmat = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])  # Doesn't matter for this test.

        n_states = tmat.shape[0]
        for state in range(n_states):
            startprob = np.zeros(n_states)
            startprob[state] = 1.0
            hsmm = DiscreteHSMMModel(
                emissions, durations, tmat, startprob=startprob
            )
            observation, sampled_state = hsmm.sample()
            self.assertEqual(sampled_state, state)
            expected_observation = np.nonzero(emissions[state])[0][0]
            self.assertEqual(observation, expected_observation)

    def test_sample_uniform_tmat(self):
        emissions = np.array([
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9],
            [0.05, 0.45, 0.45, 0.05]
        ])
        durations = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.9, 0.0, 0.0]
        ])
        tmat = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])
        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        observations, states = hsmm.sample(100000)

        # Statistics for emission probabilities.
        n_states, n_emissions = emissions.shape
        for state in range(n_states):
            mask = states == state
            num_obs = mask.sum()
            c = Counter(observations[mask])
            for emission in range(n_emissions):
                np.testing.assert_almost_equal(
                    c[emission] / float(num_obs), emissions[state, emission],
                    decimal=1
                )

        # Count durations.
        n_durations = durations.shape[1]
        states_with_durations = [
            (state, len(list(g))) for state, g in groupby(states)
        ]
        for state in range(n_states):
            counts = Counter(
                count for s, count in states_with_durations if s == state
            )
            num_counts = sum(counts.viewvalues())
            for duration in range(n_durations):
                np.testing.assert_almost_equal(
                    counts[duration + 1] / float(num_counts),
                    durations[state, duration],
                    decimal=1
                )

    def test_spiked_transition_matrix(self):
        emissions = np.array([
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9],
            [0.05, 0.45, 0.45, 0.05]
        ])
        durations = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.9, 0.0, 0.0]
        ])
        tmat = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ])
        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        observations, states = hsmm.sample(1000)

        # Check that we observe the right state transitions: 0 -> 1 -> 2 -> 0
        unique_states = np.array([s for s, _ in groupby(states)])
        expected_states = np.array(range(3) * len(states))
        expected_states = expected_states[unique_states[0]:]
        expected_states = expected_states[:len(unique_states)]
        np.testing.assert_array_equal(unique_states, expected_states)


class TestDiscreteHSMMModelDecoding(unittest.TestCase):

    def test_unambiguous_decoding(self):
        emissions = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
        ])
        durations = np.array([
            [0.1, 0.0, 0.0, 0.9],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.9, 0.0, 0.0]
        ])
        tmat = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])
        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        observations, states = hsmm.sample(1000)
        new_states = hsmm.decode(observations)
        np.testing.assert_array_equal(states, new_states)
