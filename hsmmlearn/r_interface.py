# -*- coding: utf-8 -*-
"""
Low-level rpy2 interface to the R hsmm package.

This is intended mainly for debugging: the api is not very Pythonic and closely
mimics what's available on the R side. Not everything is available yet, and one
should keep the R vignette close at hand for a description of available
parameters and return values.

"""
from __future__ import division, print_function
import numpy as np

import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, FloatVector

_HSMM = rpackages.importr("hsmm")


def _make_structure(ctor=FloatVector, **data):
    """ Create an R list out of given data items.
    """
    if not data:
        return None
    return r["list"](**{name: ctor(d) for name, d in data.items()})


def _make_vector(entries, ctor=FloatVector):
    """ Create an R vector of data.
    """
    if entries is None:
        return None
    return ctor(entries)


def _make_matrix(entries, nrow, ctor=FloatVector):
    """ Create an R matrix of data.
    """
    if entries is None:
        return None
    return r["matrix"](ctor(entries), nrow=nrow, byrow=True)


def _make_output_dict(r_structure, keys):
    return {
        key: np.asarray(r_structure.rx2(key)) for key in keys
    }


def hsmm(x, od, od_par, rd=None, rd_par=None, pi_par=None, tpm_par=None,
         M=None, Q_max=None, epsilon=None, censoring=None, prt=None,
         detailed=None, r_lim=None, p_log_lim=None, nu_lim=None):
    """ Fit a hidden semi-Markov model to given data.

    For more information about valid parameter choices, see the HSMM
    R vignette.

    Parameters that have a default of None assume the default that
    the R implementation specifies.

    Parameters
    ----------
    x : array(float), shape=(n_obs,)
        Observations
    od : str
        Type of observation distribution.
    od_par : dict
        Parameters for the observational distribution.
    rd : str
        Type of runlength distribution.
    rd_par : dict
        Parameters for the runlength distribution.
    pi_par : array, shape=(n_states,)
        Initial state distribution.
    tpm_par : array, shape=(n_states, n_states)
        Transition matrix.
    M : int
        The maximum runlength.
    Q_max : int
        The maximum number of iterations.
    epsilon : float
        Relative tolerance for successive iterations in the EM iteration
        to be considered significant.
    censoring : int
        If 1, the last visited state contributes to the likelihood.
    prt : int
        If true, print info about iterations to stdout.
    detailed : int
        If true, add parameters at each stage to the control list.
    r_lim :
        Currently not supported by the Python implementation.
    p_log_lim :
        Currently not supported by the Python implementation.
    nu_lim :
        Currently not supported by the Python implementation.

    Returns
    -------
    itr : int
        Number of iterations taken by the method.
    logl : float
        Final log likelihood.
    para_dict : dict
        Dictionary of parameters for the final system.
    ctrl_dict : dict
        Dictionary with two keys: ``solution_reached`` to indicate whether
        or not convergence was attained, and ``error``, an error indicator.
        For the precise meaning of the error codes, see the HSMM R vignette.

    """
    # FIXME This should not be too difficult to support, but I haven't had a
    # reason to do so yet.
    if r_lim is not None or p_log_lim is not None or nu_lim is not None:
        raise ValueError("r_lim, p_log_lim, nu_lim not supported (yet)")
    
    x = np.asarray(x)
    tpm_par = np.asarray(tpm_par)
    kwargs = dict(
        x=_make_vector(x),
        od=od,
        od_par=_make_structure(**od_par),
        rd=rd,
        pi_par=_make_vector(pi_par),
        tpm_par=_make_matrix(tpm_par.ravel(), tpm_par.shape[0]),
        rd_par=_make_structure(**rd_par),
        Q_max=Q_max,
        epsilon=epsilon,
        censoring=censoring,
        prt=prt,
        detailed=detailed,
        r_lim=r_lim,
        p_log_lim=p_log_lim,
        nu_lim=nu_lim
    )
    kwargs = {name: value for name, value in kwargs.items()
              if value is not None}

    fit = r["hsmm"](**kwargs)

    itr = np.asarray(fit.rx2("iter"))[0]
    logl = float(fit.rx2("logl")[0])
    para = fit.rx2("para")
    ctrl = fit.rx2("ctrl")

    ctrl_dict = {
        'solution_reached': bool(ctrl.rx2("solution.reached")[0]),
        'error': int(ctrl.rx2("error")[0])
    }

    para_dict = {
        'tpm': np.asarray(para.rx2("tpm")),
        'rd': _make_output_dict(para.rx2("rd"), rd_par.keys()),
        'od': _make_output_dict(para.rx2("od"), od_par.keys()),
    }
    
    return itr, logl, para_dict, ctrl_dict
        

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
