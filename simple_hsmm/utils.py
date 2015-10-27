import numpy as np


class NonParametricDistribution(object):

    def __init__(self, supports, probabilities):

        supports = np.atleast_1d(supports)
        probabilities = np.atleast_1d(probabilities)

        if supports.ndim > 1:
            raise ValueError("Supports array must be 1d.")
        if probabilities.ndim > 1:
            raise ValueError("Probabilities array must be 1d.")

        if supports.shape != probabilities.shape:
            raise ValueError("Supports and probabilities must have the "
                             "same dimension.")

        # Normalize probabilities
        s = probabilities.sum()
        probabilities /= s

        self._supports = supports
        self._probabilities = probabilities

        self._support_indices = -1 * np.ones(supports.max() + 1, dtype=int)
        self._support_indices[supports] = np.arange(len(supports))

    def rvs(self, size=1):
        return np.random.choice(
            self._supports, size=size, p=self._probabilities
        )

    def pmf(self, obs):
        obs_idx = self._support_indices[obs]
        assert -1 not in obs_idx  # XXXX
        return self._probabilities[obs_idx]
