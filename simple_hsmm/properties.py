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
