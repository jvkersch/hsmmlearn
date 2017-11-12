# -*- coding: utf-8 -*-
""" Low-level rpy2 interface to the R hsmm package.
"""
from __future__ import division, print_function
import numpy as np

import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, FloatVector

_HSMM = rpackages.importr("hsmm")


def _make_structure(ctor=FloatVector, **data):
    return r["list"](**{name: ctor(d) for name, d in data.items()})


def _make_vector(entries, ctor=FloatVector):
    return ctor(entries)


def _make_matrix(entries, nrow, ctor=FloatVector):
    return r["matrix"](ctor(entries), nrow=nrow)


def hsmm_sim(n, od, rd, pi_par, tpm_par, rd_par, od_par, seed=1234):
    """ Draw samples from a hidden semi-Markov model.

    For more information about valid parameter choices, see the HSMM
    R vignette.

    Parameters
    ----------
    n : int
        Number of samples to draw.
    od : str
        Type of observation distribution.
    rd : str
        Type of emission distribution.
    pi_par : array(float), shape=(n_states,)
        Initial state distribution.
    tpm_par : array(float), shape=(n_states, n_states)
        Transition matrix.
    rd_par : dict
        Parameters for the emission distribution.
    od_par : dict
        Parameters for the observational distribution.
    seed : int
        Optional seed.

    Returns
    -------
    obs : array(float), shape=(n,)
        Array of sampled observations.
    path : array(int), shape=(n,)
        Hidden states for each of the observations.

    """
    pi_par = np.asarray(pi_par)
    tpm_par = np.asarray(tpm_par)
    sim = r["hsmm.sim"](
        n=n,
        od=od,
        rd=rd,
        pi_par=_make_vector(pi_par),
        tpm_par=_make_matrix(tpm_par.ravel(), tpm_par.shape[0]),
        rd_par=_make_structure(**rd_par),
        od_par=_make_structure(**od_par),
        seed=seed)
    obs = np.asarray(sim.rx2("obs"))
    path = np.asarray(sim.rx2("path"))
    return obs, path
