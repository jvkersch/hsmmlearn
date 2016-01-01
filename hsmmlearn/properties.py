""" Data descriptors for HSMM parameters.

The parameters of a HSMM have to satisfy all sorts of properties that are
straightforward but tedious to check (e.g. the transition matrix has to be
square, the duration matrix must have just as many rows as there are states,
etc). The job of the data descriptors in this module is to do this validation,
so that the main :py:class:`HSMMModel` class doesn't have to worry about it.

Each of the descriptors checks that its arguments are valid, and then sets
one or more private attributes on the underlying :py:class:`HSMMModel`
instance with the validated argument, or quantities related to it.

"""

import numpy as np

from .emissions import AbstractEmissions


class Durations(object):
    """ Data descriptor for a durations distribution.
    """
    def __get__(self, obj, type=None):
        """ Return the durations distribution.

        Returns
        -------
        durations : numpy.ndarray, shape=(n_states, n_durations)
            Durations matrix.

        """
        return getattr(obj, '_durations', None)

    def __set__(self, obj, durations):
        """ Update the durations distribution with new durations.

        Parameters
        ----------
        obj : hsmmlearn.hsmm.HSMMModel
             The underlying HSMMModel.
        durations : numpy.ndarray or list of random variables.
             This can be either a numpy array, in which case the number of
             rows must be equal to the number of hidden states, or a list
             of scipy.stats discrete random variables. In the latter case,
             the duration probabilities are obtained directly from the random
             variables.

        """
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
    """ Data descriptor for an emissions distribution.
    """
    def __get__(self, obj, type=None):
        """ Return the emissions distribution.

        Returns
        -------
        emissions : hsmmlearn.emissions.AbstractEmissions
            The emissions distribution.

        """
        return getattr(obj, '_emissions', None)

    def __set__(self, obj, emissions):
        """ Set the emissions distribution.

        Parameters
        ----------
        obj : hsmmlearn.hsmm.HSMMModel
             The underlying HSMMModel.
        emissions : hsmmlearn.emissions.AbstractEmissions
            The emissions distribution.

        """
        if not isinstance(emissions, AbstractEmissions):
            raise TypeError(
                "Emissions parameter must be an instance of "
                "AbstractEmissions, but received an instance of {!r} "
                "instead.".format(type(emissions))
            )
        # XXX should check that emissions match with number of states.
        obj._emissions = emissions


class TransitionMatrix(object):
    """ Data descriptor for a transitions matrix.
    """

    def __get__(self, obj, type=None):
        """ Return the transition matrix.

        Returns
        -------
        tmat : numpy.ndarray, shape=(n_states, n_states)
           A transition matrix.

        """
        return getattr(obj, '_tmat', None)

    def __set__(self, obj, value):
        """ Set a new transition matrix.

        Parameters
        ----------
        obj : hsmmlearn.hsmm.HSMMModel
             The underlying HSMMModel.
        value : numpy.ndarray, shape=(n_states, n_states)
             The new transition matrix. This must be a square matrix. If
             a transition matrix was previously assigned to the HSMM,
             the new transition matrix must have the same number of rows.

        """

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
