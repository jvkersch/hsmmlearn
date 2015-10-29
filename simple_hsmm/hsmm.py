import numpy as np

from ._base import _viterbi_impl
from .properties import (
    AbstractEmissions, ContinuousEmissions, DiscreteEmissions,
    Duration, TransitionMatrix
)


class BaseHSMMModel(object):

    emssions = AbstractEmissions()

    tmat = TransitionMatrix()

    durations = Duration()

    @property
    def n_states(self):
        return self._tmat.shape[0]

    @property
    def n_durations(self):
        return self._durations.shape[1]

    def __init__(
            self, emissions, durations, tmat, startprob=None,
            support_cutoff=100):

        self.tmat = tmat

        self.support_cutoff = support_cutoff
        self.durations = durations

        self.emissions = emissions

        if startprob is None:
            startprob = np.full(self.n_states, 1.0 / self.n_states)
        self._startprob = startprob

    def decode(self, obs):
        """
        Given a series of observations, find the most likely
        internal states.

        """
        likelihoods = self._compute_likelihood(obs)

        n_obs = len(obs)
        outputs = np.empty(n_obs, dtype=np.int32)
        _viterbi_impl(
            n_obs,
            self.n_states,
            self.n_durations,
            self._durations_flat,
            self._tmat_flat,
            self._startprob,
            likelihoods.flatten(),
            outputs,
        )

        return outputs

    def sample(self, n_samples=1):
        """ Generate a random sample from the HSMM.

        """
        state = np.random.choice(self.n_states, p=self._startprob)
        duration = np.random.choice(
            self.n_durations, p=self._durations[state]) + 1

        if n_samples == 1:
            obs = self._emission_rvs[state].rvs()
            return obs, state

        states = np.empty(n_samples, dtype=int)
        observations = np.empty(n_samples)

        # Generate states array.
        state_idx = 0
        while state_idx < n_samples:
            # Adjust for right censoring (the last state may still be going on
            # when we reach the limit on the number of samples to generate).
            if state_idx + duration > n_samples:
                duration = n_samples - state_idx

            states[state_idx:state_idx+duration] = state
            state_idx += duration

            state = np.random.choice(self.n_states, p=self._tmat[state])
            duration = np.random.choice(
                self.n_durations, p=self._durations[state]
            ) + 1

        # Generate observations.
        for state in range(self.n_states):
            state_mask = states == state
            observations[state_mask] = self._emission_rvs[state].rvs(
                size=state_mask.sum(),
            )

        return observations, states


class MultinomialHSMMModel(BaseHSMMModel):
    """ A HSMM model with discrete emissions.
    """
    emissions = DiscreteEmissions()

    def decode(self, obs):
        states = super(MultinomialHSMMModel, self).decode(obs)
        return states.astype(int)

    def sample(self, n_samples=1):
        obs, states = super(MultinomialHSMMModel, self).sample(n_samples)
        if n_samples == 1:
            return int(obs), int(states)
        else:
            return obs.astype(int), states.astype(int)

    def _compute_likelihood(self, obs):
        obs = np.squeeze(obs)
        return np.vstack([rv.pmf(obs) for rv in self._emission_rvs])


class ContinuousHSMMModel(BaseHSMMModel):
    """ A HSMM model with continuous emissions.
    """
    emissions = ContinuousEmissions()

    def _compute_likelihood(self, obs):
        """
        Returns
        -------
        likelihoods : ndarray, shape=(n_states, n_observations)
            Array of per-state likelihoods.

        """
        obs = np.squeeze(obs)
        return np.vstack([rv.pdf(obs) for rv in self._emission_rvs])
