# -*- coding: utf-8 -*-
""" Low-level rpy2 interface to the R hsmm package.
"""
from __future__ import division, print_function
import numpy as np

import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, FloatVector

_HSMM = rpackages.importr("hsmm")


def _make_structure(**data):
    if not data:
        return None
    ctor = data.get('ctor', FloatVector)
    return r["list"](**{name: ctor(d) for name, d in data.items()})


def _make_vector(entries, ctor=FloatVector):
    if entries is None:
        return None
    return ctor(entries)


def _make_matrix(entries, nrow, ctor=FloatVector):
    if entries is None:
        return None
    return r["matrix"](ctor(entries), nrow=nrow, byrow=True)


def _make_output_dict(r_structure, keys):
    return {
        key: np.asarray(r_structure.rx2(key)) for key in keys
    }


def hsmm(x, od, od_par, rd=None, rd_par=None, pi_par=None, tpm_par=None,
         M=None, 
         Q_max=None,
         epsilon=None, censoring=None, prt=None, detailed=None,
         r_lim=None, p_log_lim=None, nu_lim=None):
    """ Fit a hidden semi-Markov model to given data.

    For more information about valid parameter choices, see the HSMM
    R vignette.

    Parameters
    ----------
    x : array(float), shape=(n_obs,)
        Observations
    od : str
        Type of observation distribution.
    od_par : dict
        Parameters for the observational distribution.

    Returns
    -------


    """
    x = np.asarray(x)
    tpm_par = np.asarray(tpm_par)
    kwargs = dict(
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

    fit = r["hsmm"](
        x=_make_vector(x),
        od=od,
        od_par=_make_structure(**od_par),
        **kwargs)

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
