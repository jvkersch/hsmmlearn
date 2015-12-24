import numpy as np

from .emissions import AbstractEmissions


class Durations(object):

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


class Emissions(object):
    def __get__(self, obj, type=None):
        return obj._emissions

    def __set__(self, obj, emissions):
        if not isinstance(emissions, AbstractEmissions):
            raise TypeError(
                "Emissions parameter must be an instance of "
                "AbstractEmissions, but received an instance of {!r} "
                "instead.".format(type(emissions))
            )
        # XXX should check that emissions match with number of states.
        obj._emissions = emissions


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
