from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm

from .utils import NonParametricDistribution


class AbstractEmissions(object):
    """ Base class for emissions distributions.

    To create a HSMM with a custom emission distribution, write a derived
    class that implements some (or all) of the abstract methods. If you
    don't need all of the HSMM functionality, you can get by with implementing
    only some of the methods.

    """

    __meta__ = ABCMeta

    @abstractmethod
    def sample_for_state(self, state, size=None):
        """ Return a random emission given a state.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.sample`.

        Parameters
        ----------
        state : int
            The internal state.
        size : int
            The number of random samples to generate.

        Returns
        -------
        observations : numpy.ndarray, shape=(size, )
            Random emissions.

        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, obs):
        """ Compute the likelihood of a sequence of observations.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit` and
        :py:func:`hsmmlearn.hsmm.HSMMModel.decode`.

        Parameters
        ----------
        obs : numpy.ndarray, shape=(n_obs, )
            Sequence of observations.

        Returns
        -------
        likelihood : float

        """
        raise NotImplementedError

    @abstractmethod
    def reestimate(self, gamma, obs):
        r""" Estimate the distribution parameters given sequences of
        smoothed probabilities and observations.

        The parameter ``gamma`` is an array of smoothed probabilities,
        with the entry ``gamma[s, i]`` giving the probability of
        finding the system in state ``s`` given *all* of the observations
        up to index ``i``:

        .. math::

            \gamma_{s, i} = P(s | o_1, \ldots, o_i ).

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit`.

        Parameters
        ----------
        gamma : numpy.ndarray, shape=(n_obs, )
            Smoothed probabilities.
        obs : numpy.ndarray, shape=(n_obs, )
            Observations.

        """
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        """ Make a copy of this object.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit` to
        make a copy of the emissions object before modifying it.

        """
        raise NotImplementedError


class MultinomialEmissions(AbstractEmissions):
    """ An emissions class for multinomial emissions.

    This emissions class models the case where the emissions are categorical
    variables, assuming values from 0 to some value k, and the probability
    of observing an emission given a state is modeled by a multinomial
    distribution.

    """

    # TODO this is only used by sample() and can be eliminated by inferring the
    # dtype from the generated samples.
    dtype = np.int64

    def __init__(self, probabilities):
        self._update(probabilities)

    def _update(self, probabilities):
        _probabilities = np.asarray(probabilities)
        # clip small neg residual (GH #34)
        _probabilities[_probabilities < 0] = 0

        xs = np.arange(_probabilities.shape[1])
        _probability_rvs = [
            NonParametricDistribution(xs, ps) for ps in _probabilities
        ]
        self._probabilities = _probabilities
        self._probability_rvs = _probability_rvs

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        return np.vstack([rv.pmf(obs) for rv in self._probability_rvs])

    def copy(self):
        return MultinomialEmissions(self._probabilities.copy())

    def sample_for_state(self, state, size=None):
        return self._probability_rvs[state].rvs(size=size)

    def reestimate(self, gamma, observations):
        new_emissions = np.empty_like(self._probabilities)
        for em in range(self._probabilities.shape[1]):
            mask = observations == em
            new_emissions[:, em] = (
                gamma[:, mask].sum(axis=1) / gamma.sum(axis=1)
            )
        self._update(new_emissions)


class GaussianEmissions(AbstractEmissions):
    """ An emissions class for Gaussian emissions.

    This emissions class models the case where emissions are real-valued
    and continuous, and the probability of observing an emission given
    the state is modeled by a Gaussian. The means and standard deviations
    for each Gaussian (one for each state) are stored as state on the
    class.

    """

    dtype = np.float64

    def __init__(self, means, scales):
        self.means = means
        self.scales = scales

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        # TODO: build in some check for the shape of the likelihoods, otherwise
        # this will silently fail and give the wrong results.
        return norm.pdf(obs,
                        loc=self.means[:, np.newaxis],
                        scale=self.scales[:, np.newaxis])

    def sample_for_state(self, state, size=None):
        return norm.rvs(self.means[state], self.scales[state], size)

    def copy(self):
        return GaussianEmissions(self.means.copy(), self.scales.copy())

    def reestimate(self, gamma, observations):
        p = np.sum(gamma * observations[np.newaxis, :], axis=1)
        q = np.sum(gamma, axis=1)
        new_means = p / q

        A = observations[np.newaxis, :] - new_means[:, np.newaxis]
        p = np.sum(gamma * A**2, axis=1)
        variances = p / q
        new_scales = np.sqrt(variances)

        self.means = new_means
        self.scales = new_scales
