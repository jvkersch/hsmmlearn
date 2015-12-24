from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm

from .utils import NonParametricDistribution


class AbstractEmissions(object):

    __meta__ = ABCMeta

    @abstractmethod
    def sample_for_state(self, state, size=None):
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, obs):
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def reestimate(self, gamma, obs):
        raise NotImplementedError


class MultinomialEmissions(AbstractEmissions):

    dtype = np.int64

    def __init__(self, probabilities):
        self._update(probabilities)

    def _update(self, probabilities):
        _probabilities = np.asarray(probabilities)
        xs = np.arange(_probabilities.shape[1])
        _probability_rvs = [
            NonParametricDistribution(xs, ps) for ps in probabilities
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
        """
        gamma : array, shape=(n_states, n_observations)
        observations : array, shape=(n_observations, )
        """
        new_emissions = np.empty_like(self._probabilities)
        for em in range(self._probabilities.shape[1]):
            mask = observations == em
            new_emissions[:, em] = (
                gamma[:, mask].sum(axis=1) / gamma.sum(axis=1)
            )
        self._update(new_emissions)


class GaussianEmissions(AbstractEmissions):

    dtype = np.float64

    def __init__(self, means, scales):
        self._means = means
        self._scales = scales

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        # TODO: build in some check for the shape of the likelihoods, otherwise
        # this will silently fail and give the wrong results.
        return norm.pdf(obs,
                        loc=self._means[:, np.newaxis],
                        scale=self._scales[:, np.newaxis])

    def sample_for_state(self, state, size=None):
        return norm.rvs(self._means[state], self._scales[state], size)

    def copy(self):
        return GaussianEmissions(self._means.copy(), self._scales.copy())

    def reestimate(self, gamma, observations):
        p = np.sum(gamma * observations[np.newaxis, :], axis=1)
        q = np.sum(gamma, axis=1)
        new_means = p / q

        A = observations[np.newaxis, :] - new_means[:, np.newaxis]
        p = np.sum(gamma * A**2, axis=1)
        variances = p / q
        new_scales = np.sqrt(variances)

        self._means = new_means
        self._scales = new_scales
