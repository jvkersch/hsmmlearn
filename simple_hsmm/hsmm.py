import numpy as np

from ._base import _viterbi_impl


class MultinomialHSMMModel(object):
    """ A HSMM model with discrete emissions.
    """
    def __init__(
            self, emissions, durations, tmat, startprob=None,
            support_cutoff=100):

        self._tmat = self._validate_tmat(tmat)
        self._tmat_flat = self._tmat.flatten()  # XXX .flat ?
        self._durations = self._validate_durations(durations, support_cutoff)
        self._durations_flat = self._durations.flatten()
        self._emissions = self._validate_emissions(emissions)

        if startprob is None:
            startprob = np.full(self.n_states, 1.0 / self.n_states)
        self._startprob = startprob

    @property
    def n_states(self):
        return self._tmat.shape[0]

    def _validate_tmat(self, tmat):
        tmat = np.asarray(tmat)
        if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
            raise ValueError("Transition matrix must be square, "
                             "but a matrix of shape {0} was received.".format(
                                 tmat.shape))
        return tmat

    def _validate_durations(self, durations, support_cutoff):
        if isinstance(durations, np.ndarray):
            durations = np.asarray(durations)
            if durations.ndim != 2 or durations.shape[0] != self.n_states:
                msg = "Duration matrix must be 2d and have {0} rows.".format(
                    self._tmat.shape[0]
                )
                raise ValueError(msg)
            return durations
        else:
            if len(durations) != self.n_states:
                raise ValueError("The 'durations' parameters must have "
                                 "length {}.".format(self.n_states))

            support = np.arange(1, support_cutoff + 1)
            durations_array = np.empty((self.n_states, support_cutoff))
            for k, rv in enumerate(durations):
                durations_array[k] = rv.pmf(support)
            durations_array /= durations_array.sum(axis=1)[:, np.newaxis]

            return durations_array

    def _validate_emissions(self, emissions):
        # For now, just discrete emssions are supported.
        emissions = np.asarray(emissions)
        if emissions.ndim != 2 or emissions.shape[0] != self.n_states:
            msg = "Emission matrix must be 2d and have {0} rows.".format(
                self.n_states
            )
            raise ValueError(msg)
        return emissions

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


class ContinuousHSMMModel(object):
    """ A HSMM model with continuous emissions.
    """
    def __init__(
            self, emission_rv, emission_loc, emission_scale,
            durations, tmat, startprob=None,
            support_cutoff=100):

        self._tmat = self._validate_tmat(tmat)
        self._tmat_flat = self._tmat.flatten()  # XXX .flat ?
        self._durations = self._validate_durations(durations, support_cutoff)
        self._durations_flat = self._durations.flatten()

        # XXX validate this TODO Internal, but should become part of the api:
        # freeze emission rvs according to locations and scales.
        
        #self._emission_rv = emission_rv
        #self._emission_loc = emission_loc
        #self._emission_scale = emission_scale

        self._emission_rvs = [
            emission_rv(loc=loc, scale=scale)
            for (loc, scale) in zip(emission_loc, emission_scale)
        ]

        if startprob is None:
            startprob = np.full(self.n_states, 1.0 / self.n_states)
        self._startprob = startprob

    @property
    def n_states(self):
        return self._tmat.shape[0]

    def _validate_tmat(self, tmat):
        tmat = np.asarray(tmat)
        if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
            raise ValueError("Transition matrix must be square, "
                             "but a matrix of shape {0} was received.".format(
                                 tmat.shape))
        return tmat

    def _validate_durations(self, durations, support_cutoff):
        if isinstance(durations, np.ndarray):
            durations = np.asarray(durations)
            if durations.ndim != 2 or durations.shape[0] != self.n_states:
                msg = "Duration matrix must be 2d and have {0} rows.".format(
                    self._tmat.shape[0]
                )
                raise ValueError(msg)
            return durations
        else:
            if len(durations) != self.n_states:
                raise ValueError("The 'durations' parameters must have "
                                 "length {}.".format(self.n_states))

            support = np.arange(1, support_cutoff + 1)
            durations_array = np.empty((self.n_states, support_cutoff))
            for k, rv in enumerate(durations):
                durations_array[k] = rv.pmf(support)
            durations_array /= durations_array.sum(axis=1)[:, np.newaxis]

            return durations_array

    def _compute_likelihood(self, obs):
        """
        Returns
        -------
        likelihoods : ndarray, shape=(n_states, n_observations)
            Array of per-state likelihoods.

        """
        obs = np.squeeze(obs)
        return np.vstack([rv.pdf(obs) for rv in self._emission_rvs])

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
        #n_states, n_emissions = self._emissions.shape
        n_durations = self._durations.shape[1]

        state = np.random.choice(self.n_states, p=self._startprob)
        duration = np.random.choice(n_durations, p=self._durations[state]) + 1

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
                n_durations, p=self._durations[state]
            ) + 1

        # Generate observations.
        for state in range(self.n_states):
            state_mask = states == state
            observations[state_mask] = self._emission_rvs[state].rvs(
                size=state_mask.sum(),
            )

        return observations, states
