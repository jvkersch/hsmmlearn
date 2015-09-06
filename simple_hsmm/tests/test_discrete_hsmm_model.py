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


class TestProperties(unittest.TestCase):

    def test_set_initial_transmat(self):
        emissions = np.zeros((3, 5))
        durations = np.zeros((3, 7))
        tmat = np.eye(3)

        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        np.testing.assert_array_equal(hsmm._tmat, tmat)
        np.testing.assert_array_equal(hsmm._tmat_flat, tmat.flatten())

        self.assertEqual(hsmm.n_states, 3)

        # Non-square tmat
        tmat = np.ones((3, 4))
        with self.assertRaisesRegexp(ValueError, "shape \(3, 4\)"):
            hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        # Non-matrix tmat
        tmat = np.ones((3, 4, 1))
        with self.assertRaisesRegexp(ValueError, "shape \(3, 4, 1\)"):
            hsmm = DiscreteHSMMModel(emissions, durations, tmat)

    def test_reassign_transmat(self):
        emissions = np.zeros((3, 5))
        durations = np.zeros((3, 7))
        tmat = np.eye(3)

        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        new_tmat = 2 * tmat
        hsmm.transmat = new_tmat

        self.assertEqual(hsmm.n_states, 3)
        np.testing.assert_array_equal(hsmm._tmat, new_tmat)
        np.testing.assert_array_equal(hsmm._tmat_flat, new_tmat.flatten())

        new_tmat = np.eye(4)
        with self.assertRaisesRegexp(ValueError, "shape \(4, 4\)"):
            hsmm.transmat = new_tmat

    def test_emissions(self):
        emissions = np.arange(3 * 5).reshape(3, 5)
        durations = np.zeros((3, 7))
        tmat = np.eye(3)

        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        np.testing.assert_array_equal(hsmm._emissions, emissions)

        # Non-matrix tmat
        emissions = emissions.reshape(3, 5, 1)
        with self.assertRaisesRegexp(ValueError, "be 2d"):
            hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        # Emissions matrix with too many states
        emissions = np.zeros((4, 4))
        with self.assertRaisesRegexp(ValueError, "3 rows"):
            hsmm = DiscreteHSMMModel(emissions, durations, tmat)

    def test_durations(self):
        durations = np.arange(3 * 5).reshape(3, 5)
        emissions = np.zeros((3, 7))
        tmat = np.eye(3)

        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        np.testing.assert_array_equal(hsmm._durations, durations)
        np.testing.assert_array_equal(
            hsmm._durations_flat, durations.flatten()
        )

        # Non-matrix tmat
        durations = durations.reshape(3, 5, 1)
        with self.assertRaisesRegexp(ValueError, "be 2d"):
            hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        # Durations matrix with too many states
        durations = np.zeros((4, 4))
        with self.assertRaisesRegexp(ValueError, "3 rows"):
            hsmm = DiscreteHSMMModel(emissions, durations, tmat)

    def test_startprob_implicit(self):
        durations = np.arange(3 * 5).reshape(3, 5)
        emissions = np.zeros((3, 7))
        tmat = np.eye(3)

        hsmm = DiscreteHSMMModel(emissions, durations, tmat)

        np.testing.assert_array_equal(
            hsmm.startprob, np.full(tmat.shape[0], 1.0 / 3)
        )

    def test_startprob_explicit(self):
        durations = np.arange(3 * 5).reshape(3, 5)
        emissions = np.zeros((3, 7))
        tmat = np.eye(3)
        startprob = np.arange(3)

        hsmm = DiscreteHSMMModel(
            emissions, durations, tmat, startprob=startprob
        )

        np.testing.assert_array_equal(hsmm.startprob, startprob)

        startprob = np.arange(4)
        with self.assertRaisesRegexp(ValueError, "have 3 elements"):
            hsmm.startprob = startprob

        startprob = np.eye(3)
        with self.assertRaisesRegexp(ValueError, "be 1d"):
            hsmm.startprob = startprob
