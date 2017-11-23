# -*- coding: utf-8 -*-
from __future__ import division, print_function

import unittest

import numpy as np
import numpy.testing as nptest

try:
    from hsmmlearn.r_interface import hsmm, hsmm_sim
    HAS_R_SUPPORT = True
except ImportError:
    HAS_R_SUPPORT = False

skip_no_r = unittest.skipIf(not HAS_R_SUPPORT, "No R support")


@skip_no_r
class TestHSMM(unittest.TestCase):

    def test_r_data(self):
        # Given
        n = 2000
        od = "norm"
        rd = "log"
        pi_par = [1.0/3]*3
        tpm_par = np.array([[0, 0.5, 0.5], [0.7, 0, 0.3], [0.8, 0.2, 0]])
        rd_par = {'p': [0.98, 0.98, 0.99]}
        od_par = {'mean': [-1.5, 0.0, 1.5], 'var': [0.5, 0.6, 0.8]}
        seed = 3539
        obs, _ = hsmm_sim(n, od, rd, pi_par, tpm_par, rd_par, od_par, seed)

        # When
        itr, logl, para, ctrl = hsmm(
            obs, od, od_par, rd, rd_par, pi_par, tpm_par
        )

        # Then
        self.assertTrue(ctrl['solution_reached'])
        self.assertEqual(ctrl['error'], 0)
        self.assertEqual(itr, 28)
        self.assertAlmostEqual(logl, -2671.790242590307)

        nptest.assert_allclose(
            para['tpm'], np.array([
                [0.0000000, 0.5191888, 0.4808112],
                [0.8148501, 0.0000000, 0.1851499],
                [0.8951527, 0.1048473, 0.0000000]]),
            rtol=1e-5, atol=1e-5
        )
        nptest.assert_allclose(
            para['rd']['p'], np.array([0.9881984, 0.9588898, 0.9905475]),
            rtol=1e-5, atol=1e-5)
        nptest.assert_allclose(
            para['od']['mean'], np.array([-1.54168985, 0.08857843, 1.52942672]),
            rtol=1e-5, atol=1e-5)
        nptest.assert_allclose(
            para['od']['var'], np.array([0.4780348, 0.6139798, 0.7984119]),
            rtol=1e-5, atol=1e-5)


@skip_no_r
class TestHSMMSim(unittest.TestCase):

    def test_r_data(self):
        # Given
        n = 10
        od = "norm"
        rd = "log"
        pi_par = [1.0/3]*3
        tpm_par = np.array([[0, 0.5, 0.5], [0.8, 0, 0.2], [0.7, 0.3, 0]])
        rd_par = {'p': [0.98, 0.98, 0.99]}
        od_par = {'mean': [-1.5, 0, 1.5], 'var': [0.5, 0.6, 0.8]}
        seed = 1234

        # When
        obs, path = hsmm_sim(n, od, rd, pi_par, tpm_par, rd_par, od_par, seed)

        # Then
        nptest.assert_allclose(
            obs,
            np.array([0.21489577, 0.84000452, -1.81696963, 0.33239855,
                      0.39198921, -0.44519166, -0.42341921, -2.12935178,
                      -1.83742619, -0.60128368])
        )
        nptest.assert_array_equal(
            path, np.array([2, 2, 2, 2, 2, 2, 2, 1, 1, 2])
        )


