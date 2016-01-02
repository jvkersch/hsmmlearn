import numpy as np

from .base import _viterbi_impl, _fb_impl
from .properties import Durations, Emissions, TransitionMatrix


class NoConvergenceError(Exception):
    pass


class HSMMModel(object):
    """HSMM model base class.

    This class provides the basic functionality to work with hidden semi-Markov
    models:

    #. Sampling from a HSMM via the :py:func:`sample` method;
    #. Fitting the parameters of a HSMM to a sequence of observations via the
       :py:func:`fit` method.

    These methods are agnostic as to what HSMM is being used: the specifics of
    the model are all encapsulated in the properties `tmat`, `emissions`, and
    `durations`, which control the transition matrix, the emission
    distribution, and the duration distribution. In particular, the emission
    distribution can be discrete or continuous.

    """

    #: Data descriptor for the transition matrix.
    tmat = TransitionMatrix()

    #: Duration distribution matrix data descriptor. Each row corresponds to
    #: a hidden state, with the `i`-th entry the probability of seeing duration
    #: `i` in that state.
    durations = Durations()

    #: Emission distribution data descriptor. Valid assignments to this
    #: descriptor must be a subclass of
    #: :py:class:`hsmmlearn.emissions.AbstractEmissions`.
    emissions = Emissions()

    @property
    def n_states(self):
        """ The number of hidden states for this HSMM.
        """
        return self._tmat.shape[0]

    @property
    def n_durations(self):
        """ The number of durations for this HSMM.
        """
        return self._durations.shape[1]

    def __init__(
            self, emissions, durations, tmat, startprob=None,
            support_cutoff=100):
        """ Create a new HSMM instance.

        Parameters
        ----------
        emissions : hsmmlearn.emissions.AbstractEmissions
            Emissions distribution to use.
        durations : numpy.ndarray or list of random variables.
            Durations matrix with shape (n_states, n_durations). If a list of
            random variables is passed in, all RVs must be subclasses of
            ``scipy.stats.rv_discrete``. In this case, the duration
            probabilities are obtained from the support of the RVs, from
            1 to `support_cutoff`.
        tmat : numpy.ndarray, shape=(n_states, n_states)
            Transition matrix.
        startprob : numpy.ndarray, shape=(n_states, )
            Initial probabilities for the Markov chain. If this is ``None``,
            the uniform distribution is assumed (all states are equally
            likely).
        support_cutoff : int
            Maximal duration to take into account. This is used when passing in
            a list of random variables for the durations, which will then be
            sampled from 1 to ``support_cutoff``.

        """
        # TODO: move support_cutoff parameter to the durations property.
        self.tmat = tmat

        self.support_cutoff = support_cutoff
        self.durations = durations

        self.emissions = emissions

        if startprob is None:
            startprob = np.full(self.n_states, 1.0 / self.n_states)
        self._startprob = startprob

    def decode(self, obs):
        """ Find most likely internal states for a sequence of observations.

        Given a sequence of observations, this method runs the Viterbi
        algorithm to find the most likely sequence of corresponding internal
        states. "Most likely" should be interpreted here (as in the classical
        Viterbi algorithm) as the sequence of states that maximizes the joint
        probability P(observations, states).

        Parameters
        ----------
        obs : numpy.ndarray, shape=(n_obs, )
            Observations.

        Returns
        -------
        states : numpy.ndarray, shape=(n_obs, )
            Reconstructed internal states.

        """
        likelihoods = self.emissions.likelihood(obs)

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

        Parameters
        ----------
        n_samples : int
             Number of samples to generate.

        Returns
        -------
        observations, states : numpy.ndarray, shape=(n_samples, )
            Random sample of observations and internal states.

        """
        state = np.random.choice(self.n_states, p=self._startprob)
        duration = np.random.choice(
            self.n_durations, p=self._durations[state]) + 1

        if n_samples == 1:
            obs = self.emissions.sample_for_state(state)
            return obs, state

        states = np.empty(n_samples, dtype=int)
        observations = np.empty(n_samples, dtype=self.emissions.dtype)

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
            observations[state_mask] = self.emissions.sample_for_state(
                state, size=state_mask.sum(),
            )

        return observations, states

    def fit(self, obs, max_iter=20, atol=1e-5, censoring=True):
        """ Fit the parameters of a HSMM to a given sequence of observations.

        This method runs the expectation-maximization algorithm to adjust the
        parameters of the HSMM to fit a given sequence of observations as well
        as possible. The HSMM will be updated in place, unless an error occurs,
        in which case the original HSMM is restored.

        Parameters
        ----------
        obs : numpy.ndarray, shape=(n_samples, )
            Sequence of observations.
        max_iter : int
            Maximum number of EM steps to do before terminating.
        atol : float
            Absolute tolerance to decide whether the EM algorithm has
            converged.
        censoring : bool
            Whether to apply right-censoring.

        Raises
        ------
        NoConvergenceError
            If the algorithm terminated abnormally before converging.

        """
        obs = np.atleast_1d(obs)

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
            likelihoods = self.emissions.likelihood(obs)

            likelihoods[likelihoods < 1e-12] = 1e-12

            err = _fb_impl(
                censoring, tau, j, m,
                durations_flat, tmat_flat, startprob,
                likelihoods.reshape(-1),
                f.reshape(-1), l.reshape(-1), g.reshape(-1), l1.reshape(-1),
                n, norm.reshape(-1), eta.reshape(-1), xi.reshape(-1)
            )
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

            # Re-estimate emissions
            self.emissions.reestimate(l, obs)

            # Reassign!
            self.tmat = new_tmat
            self.durations = new_durations
            self._startprob = new_pi

        if err != 0:
            # An error occurred.
            self.emissions = old_emissions
            self.tmat = old_tmat
            self.durations = old_durations
            self._startprob = old_startprob
            raise NoConvergenceError(
                "The forward-backward algorithm encountered an internal error "
                "after {} steps. Try reducing the `num_iter` parameter. "
                "Log-likelihood procession: {}.".format(step, log_likelihoods))

        return has_converged, llh
