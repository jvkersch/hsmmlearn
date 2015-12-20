import numpy as np

from .utils import NonParametricDistribution


# XXX Should descriptors raise AttributeError instead?

class Duration(object):

    def __get__(self, obj, type=None):
        return obj._durations

    def __set__(self, obj, durations):
        if isinstance(durations, np.ndarray):
            durations_arr = np.asarray(durations)
            if (durations_arr.ndim != 2 or
                    durations_arr.shape[0] != obj.n_states):
                msg = "Duration matrix must be 2d and have {0} rows.".format(
                    obj.n_states
                )
                raise ValueError(msg)
        else:
            if len(durations) != obj.n_states:
                raise ValueError("The 'durations' parameters must have "
                                 "length {}.".format(obj.n_states))

            support = np.arange(1, obj.support_cutoff + 1)
            durations_arr = np.empty((obj.n_states, obj.support_cutoff))
            for k, rv in enumerate(durations):
                durations_arr[k] = rv.pmf(support)
            durations_arr /= durations_arr.sum(axis=1)[:, np.newaxis]

        obj._durations = durations_arr
        obj._durations_flat = durations_arr.flatten()


class TransitionMatrix(object):

    def __get__(self, obj, type=None):
        return obj._tmat

    def __set__(self, obj, value):
        tmat = self._validate_tmat(value)
        if hasattr(obj, '_tmat'):
            if obj._tmat.shape != tmat.shape:
                raise ValueError("New transition matrix has {} states, "
                                 "which is incompatible with old transition "
                                 "matrix, which has {} states.".format(
                                     tmat.shape[0], obj._tmat.shape[0]))

        obj._tmat = tmat
        obj._tmat_flat = tmat.flatten()

    def _validate_tmat(self, tmat):
        tmat = np.asarray(tmat)
        if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
            raise ValueError("Transition matrix must be square, "
                             "but a matrix of shape {0} was received.".format(
                                 tmat.shape))
        return tmat


class AbstractEmissions(object):

    def __get__(self, obj, type=None):
        raise AttributeError("This property should not be used directly.")

    def __set__(self, obj, value):
        raise AttributeError("This property should not be used directly.")


class ContinuousEmissions(AbstractEmissions):

    def __get__(self, obj, type=None):
        return obj._emission_rvs

    def __set__(self, obj, emission_rvs):
        # TODO Validation not yet implemented.
        obj._emission_rvs = emission_rvs


class GaussianEmissions(ContinuousEmissions):

    def __get__(self, obj, type=None):

        return obj._emission_rvs

    def __set__(self, obj, value):
        from scipy.stats import norm
        obj._means = value[0]
        obj._scales = value[1]
        obj._emission_rvs = [norm(loc=loc, scale=scale)
                             for (loc, scale) in zip(obj._means, obj._scales)]

    @classmethod
    def re_estimate(cls, old_emissions, ggamma, observations):
        """ Re-estimate the parameters of this distribution, given a set of
        smoothed probabilities.

        In Guedon's paper, these quantities are called L_j(t).

        Parameters
        ----------
        gamma : array, shape=(n_states, n_observations)
        observations : array, shape=(n_observations, )

        """
        p = np.sum(gamma * observations[np.newaxis, :], axis=1)
        q = np.sum(gamma, axis=1)
        new_means = p / q

        A = observations[np.newaxis, :] - new_means[:, np.newaxis]
        p = np.sum(gamma * A**2, axis=1)
        variances = p / q
        new_scales = np.sqrt(variances)

        return new_means, new_scales


class DiscreteEmissions(AbstractEmissions):

    def __get__(self, obj, type=None):
        return obj._emissions

    def __set__(self, obj, emissions):
        emissions = np.asarray(emissions)
        if emissions.ndim != 2 or emissions.shape[0] != obj.n_states:
            msg = "Emission matrix must be 2d and have {0} rows.".format(
                obj.n_states
            )
            raise ValueError(msg)

        xs = np.arange(emissions.shape[1])
        _emission_rvs = [
            NonParametricDistribution(xs, ps) for ps in emissions
        ]

        obj._emissions = emissions
        obj._emission_rvs = _emission_rvs

    @classmethod
    def re_estimate(cls, old_emissions, gamma, observations):
        """
        gamma : array, shape=(n_states, n_observations)
        observations : array, shape=(n_observations, )
        """
        new_emissions = np.empty_like(old_emissions)
        for em in range(old_emissions.shape[1]):
            mask = observations == em
            new_emissions[:, em] = (
                gamma[:, mask].sum(axis=1) / gamma.sum(axis=1)
            )
        return new_emissions
