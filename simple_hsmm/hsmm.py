import numpy as np

from ._base import _viterbi_impl


class DiscreteHSMMModel(object):
    """ A HSMM model for which both the durations and emissions are
    discretely distributed.

    """
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
        self.transmat = tmat
        self.emissions = emissions
        self.durations = durations

        if startprob is None:
            startprob = np.full(self.n_states, 1.0 / self.n_states)
        self.startprob = startprob

    @property
    def n_states(self):
        return self._tmat.shape[0]

    @property
    def transmat(self):
        return self._tmat

    @transmat.setter
    def transmat(self, value):
        tmat = np.asarray(value)
        if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
            raise ValueError("Transition matrix must be square, "
                             "but a matrix of shape {0} was received.".format(
                                 tmat.shape))

        if hasattr(self, '_tmat') and self._tmat.shape[0] != tmat.shape[0]:
            raise ValueError(
                ("Shape {0} of new transition matrix differs from "
                 "previous shape {1}.").format(self._tmat.shape, tmat.shape)
            )

        # TODO Normalize transition matrix
        self._tmat = tmat
        self._tmat_flat = tmat.flatten()

    @property
    def emissions(self):
        return self._emissions

    @emissions.setter
    def emissions(self, value):
        emissions = np.asarray(value)
        if emissions.ndim != 2 or emissions.shape[0] != self.n_states:
            msg = "Emission matrix must be 2d and have {0} rows.".format(
                self.n_states
            )
            raise ValueError(msg)

        self._emissions = emissions

    @property
    def durations(self):
        return self._durations

    @durations.setter
    def durations(self, value):
        durations = np.asarray(value)
        if durations.ndim != 2 or durations.shape[0] != self.n_states:
            msg = "Duration matrix must be 2d and have {0} rows.".format(
                self.n_states
            )
            raise ValueError(msg)

        self._durations = durations
        self._durations_flat = durations.flatten()

    @property
    def startprob(self):
        return self._startprob

    @startprob.setter
    def startprob(self, value):
        startprob = np.asarray(value)
        if startprob.ndim != 1 or startprob.shape[0] != self.n_states:
            msg = ("Starting probabilities must be 1d and have {0} "
                   "elements.").format(self.n_states)
            raise ValueError(msg)
        self._startprob = startprob

    def _compute_likelihood(self, obs):
        return self._emissions[:, obs]

    def decode(self, obs):
        """
        Given a series of observations, find the most likely
        internal states.

        """
        likelihoods = self._compute_likelihood(obs)

        n_durations = self._durations.shape[1]
        n_states, n_obs = likelihoods.shape

        outputs = np.empty(n_obs, dtype=np.int32)
        _viterbi_impl(
            n_obs,
            n_states,
            n_durations,
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
        n_states, n_emissions = self._emissions.shape
        n_durations = self._durations.shape[1]

        state = np.random.choice(n_states, p=self._startprob)
        duration = np.random.choice(n_durations, p=self._durations[state]) + 1

        if n_samples == 1:
            observation = np.random.choice(
                n_emissions, p=self._emissions[state]
            )
            return observation, state

        states = np.empty(n_samples, dtype=int)
        observations = np.empty(n_samples, dtype=int)

        # Generate states array.
        state_idx = 0
        while state_idx < n_samples:
            # Adjust for right censoring (the last state may still be going on
            # when we reach the limit on the number of samples to generate).
            if state_idx + duration > n_samples:
                duration = n_samples - state_idx

            states[state_idx:state_idx+duration] = state
            state_idx += duration

            state = np.random.choice(n_states, p=self._tmat[state])
            duration = np.random.choice(
                n_durations, p=self._durations[state]
            ) + 1

        # Generate observations.
        for state in range(n_states):
            state_mask = states == state
            observations[state_mask] = np.random.choice(
                n_emissions, size=state_mask.sum(), p=self._emissions[state]
            )

        return observations, states
