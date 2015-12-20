import numpy as np

from ._base import _viterbi_impl, _fb_impl
from .properties import (
    AbstractEmissions, ContinuousEmissions, DiscreteEmissions,
    GaussianEmissions, Duration, TransitionMatrix
)


class NoConvergenceError(Exception):
    pass


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

    def _get_reestimator(self):
        cls = type(self).__dict__['emissions'].__class__
        return cls.re_estimate


class DiscreteHSMMModel(BaseHSMMModel):
    """ A HSMM model with discrete emissions.
    """
    emissions = DiscreteEmissions()

    def decode(self, obs):
        states = super(DiscreteHSMMModel, self).decode(obs)
        return states.astype(int)

    def sample(self, n_samples=1):
        obs, states = super(DiscreteHSMMModel, self).sample(n_samples)
        if n_samples == 1:
            return int(obs), int(states)
        else:
            return obs.astype(int), states.astype(int)

    def _compute_likelihood(self, obs):
        obs = np.squeeze(obs)
        return np.vstack([rv.pmf(obs) for rv in self._emission_rvs])

    def fit(self, obs, max_iter=1, atol=1e-5, censoring=True):
        """Fit HSMM parameters (in place), given a sequence of observations.

        """

        censoring = int(censoring)
        tau = len(obs)
        j = self.n_states
        m = self.n_durations

        f = np.zeros((j, tau))
        l = np.zeros((j, tau))
        g = np.zeros((j, tau))
        l1 = np.zeros((j, tau))
        n = np.zeros(tau)
        norm = np.zeros((j, tau))
        eta = np.zeros((j, m))
        xi = np.zeros((j, m))

        log_likelihoods = []

        llh = -np.inf
        has_converged = False

        # Make a copy of HSMM elements, so that we can recover in case of
        # error.
        old_tmat = self.tmat.copy()
        old_durations = self.durations.copy()
        old_emissions = self.emissions.copy()
        old_startprob = self._startprob.copy()

        # TODO Make startprob into a property like the rest.

        for step in range(1, max_iter + 1):
            durations_flat = self._durations_flat.copy()
            tmat_flat = self._tmat_flat.copy()
            startprob = self._startprob.copy()
            likelihoods = self._compute_likelihood(obs)

            likelihoods[likelihoods < 1e-12] = 1e-12

            err = _fb_impl(
                censoring, tau, j, m,
                durations_flat, tmat_flat, startprob,
                likelihoods.reshape(-1),
                f.reshape(-1), l.reshape(-1), g.reshape(-1), l1.reshape(-1),
                n, norm.reshape(-1), eta.reshape(-1), xi.reshape(-1)
            )
            print err
            if err != 0:
                break

            # Calculate log likelihood
            new_llh = np.log(n).sum()
            log_likelihoods.append(new_llh)
            if np.abs(new_llh - llh) < atol:
                has_converged = True
                break
            llh = new_llh

            # Re-estimate pi
            new_pi = l[:, 0]

            # Re-estimate tmat
            new_tmat = np.empty_like(self._tmat)
            for i in range(j):
                r = l1[i, :-1].sum()
                for k in range(j):
                    z = self._tmat[i, k] * (g[k, 1:] * f[i, :-1]).sum()
                    new_tmat[i, k] = z / r

            # Re-estimate durations
            denominator = l1[:, :-1].sum(axis=1)
            if censoring:
                denominator += l[:, -1]
            new_durations = eta / denominator[:, np.newaxis]

            # Re-estimate discrete emissions
            re_estimate = self._get_reestimator()
            new_emissions = re_estimate(self.emissions, l, obs)

            # Reassign!
            self.tmat = new_tmat
            self.durations = new_durations
            self.emissions = new_emissions
            self._startprob = new_pi

        if err != 0:
            # An error occurred.
            self.tmat = old_tmat
            self.emissions = old_emissions
            self.durations = old_durations
            self._startprob = old_startprob
            raise NoConvergenceError(
                "The forward-backward algorithm encountered an internal error "
                "after {} steps. Try reducing the `num_iter` parameter. "
                "Log-likelihood procession: {}.".format(step, log_likelihoods))

        return has_converged, llh


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


class GaussianHSMMModel(ContinuousHSMMModel):

    emissions = GaussianEmissions()

    def __init__(
            self, means, scales, durations, tmat, startprob=None,
            support_cutoff=100):

        self.tmat = tmat

        self.support_cutoff = support_cutoff
        self.durations = durations

        #self.emissions = emissions
        self.emissions = (means, scales)

        if startprob is None:
            startprob = np.full(self.n_states, 1.0 / self.n_states)
        self._startprob = startprob

    def fit(self, obs, max_iter=20, atol=1e-5, censoring=True):
        """Fit HSMM parameters (in place), given a sequence of observations.

        """
        censoring = int(censoring)
        tau = len(obs)
        j = self.n_states
        m = self.n_durations

        f = np.zeros((j, tau))
        l = np.zeros((j, tau))
        g = np.zeros((j, tau))
        l1 = np.zeros((j, tau))
        n = np.zeros(tau)
        norm = np.zeros((j, tau))
        eta = np.zeros((j, m))
        xi = np.zeros((j, m))

        log_likelihoods = []

        llh = -np.inf
        has_converged = False

        # Make a copy of HSMM elements, so that we can recover in case of
        # error.
        old_tmat = self.tmat.copy()
        old_durations = self.durations.copy()
        old_means = self._means.copy()
        old_scales = self._scales.copy()
        # old_emissions = self.emissions.copy()
        old_startprob = self._startprob.copy()

        # TODO Make startprob into a property like the rest.

        for step in range(1, max_iter + 1):
            durations_flat = self._durations_flat.copy()
            tmat_flat = self._tmat_flat.copy()
            startprob = self._startprob.copy()
            likelihoods = self._compute_likelihood(obs)

            err = _fb_impl(
                censoring, tau, j, m,
                durations_flat, tmat_flat, startprob,
                likelihoods.reshape(-1),
                f.reshape(-1), l.reshape(-1), g.reshape(-1), l1.reshape(-1),
                n, norm.reshape(-1), eta.reshape(-1), xi.reshape(-1)
            )

            # Calculate log likelihood
            new_llh = np.log(n).sum()
            log_likelihoods.append(new_llh)
            if np.abs(new_llh - llh) < atol:
                has_converged = True
                break
            llh = new_llh

            # Re-estimate pi
            new_pi = l[:, 0]
            new_pi[new_pi < 1e-14] = 1e-14
            new_pi /= new_pi.sum()

            # Re-estimate tmat
            new_tmat = np.empty_like(self._tmat)
            for i in range(j):
                r = l1[i, :-1].sum()
                for k in range(j):
                    z = self._tmat[i, k] * (g[k, 1:] * f[i, :-1]).sum()
                    new_tmat[i, k] = z / r

            # Re-estimate durations
            denominator = l1[:, :-1].sum(axis=1)
            if censoring:
                denominator += l[:, -1]
            new_durations = eta / denominator[:, np.newaxis]

            # Re-estimate discrete emissions
            re_estimate = self._get_reestimator()
            new_means, new_scales = re_estimate(self.emissions, l, obs)

            # Reassign!
            self.tmat = new_tmat
            self.durations = new_durations
            self.emissions = (new_means, new_scales)
            self._startprob = new_pi

        if err != 0:
            # An error occurred.
            self.tmat = old_tmat
            self.emissions = (old_means, old_scales)
            self.durations = old_durations
            self._startprob = old_startprob
            raise NoConvergenceError(
                "The forward-backward algorithm encountered an internal error "
                "after {} steps. Try reducing the `num_iter` parameter. "
                "Log-likelihood procession: {}.".format(step, log_likelihoods))

        return has_converged, llh
