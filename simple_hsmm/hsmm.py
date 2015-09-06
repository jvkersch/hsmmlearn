import numpy as np

from ._base import _viterbi_impl


class DiscreteHSMMModel(object):
    def __init__(self, emissions, durations, tmat, startprob=None):
        """
        Parameters
        ----------
        emissions : array, shape=(n_states, n_emissions)
            Discrete emission distribution.
        durations : array, shape=(n_states, n_durations)
            Duration distribution per state.
        tmat : array, shape=(n_states, n_states)
            Transition matrix.
        startprob : array, shape=(n_states, )
            Initial probabilities for the hidden states. If this is None,
            the uniform distribution will be used.

        """
        self._emissions = emissions
        self._durations = durations
        self._tmat = tmat

        n_states = tmat.shape[0]
        if startprob is None:
            startprob = np.full(n_states, 1.0 / n_states)
        self._startprob = startprob

    def _compute_likelihood(self, obs):
        return self._emissions[:, obs]

    def decode(self, obs):
        likelihoods = self._compute_likelihood(obs)

        n_durations = self._durations.shape[1]
        n_states, n_obs = likelihoods.shape

        outputs = np.empty(n_obs, dtype=np.int32)
        _viterbi_impl(
            n_obs,
            n_states,
            n_durations,
            self._durations.flatten(),  # TODO cache this in a property
            self._tmat.flatten(),
            self._startprob,
            likelihoods.flatten(),
            outputs,
        )

        return outputs

    def sample(self, n_samples=1):
        n_states, n_emissions = self._emissions.shape
        n_durations = self._durations.shape[1]

        state = np.random.choice(n_states, p=self._startprob)
        observation = np.random.choice(n_emissions, p=self._emissions[state])
        duration = np.random.choice(n_durations, p=self._durations[state]) + 1

        if n_samples == 1:
            return observation, state
        else:
            states = np.empty(n_samples, dtype=int)
            observations = np.empty(n_samples, dtype=int)

            n_generated = 0
            states[n_generated] = state
            observations[n_generated] = observation
            n_generated += 1
            duration -= 1

            while True:
                while duration > 0 and n_generated < n_samples:
                    observation = np.random.choice(
                        n_emissions, p=self._emissions[state]
                    )
                    states[n_generated] = state
                    observations[n_generated] = observation
                    n_generated += 1
                    duration -= 1

                if n_generated >= n_samples:
                    break

                state = np.random.choice(n_states, p=self._tmat[state])
                duration = np.random.choice(
                    n_durations, p=self._durations[state]) + 1

            return observations, states
