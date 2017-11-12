# -*- coding: utf-8 -*-
from __future__ import division, print_function

import unittest

import numpy as np
import numpy.testing as nptest

from hsmmlearn.r_interface import hsmm_sim


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
