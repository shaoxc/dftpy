#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functional import Functional
from dftpy.formats import io


class Test(unittest.TestCase):
    def test_gga(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        rho_r = io.read_density(dftpy_data_path + "/Al_fde_rho.pp")
        optional_kwargs_gga = {}
        optional_kwargs_gga['k_str'] = 'lc94'
        thefuncclass = Functional(type='KEDF',
                                  name='GGA',
                                  **optional_kwargs_gga)
        func = thefuncclass(rho=rho_r)
        self.assertTrue(np.isclose(func.energy, 1.6821337114254904))
        self.assertTrue(np.isclose((func + func).energy, 1.6821337114254904 * 2))
        self.assertTrue(np.isclose((func * 2).energy, 1.6821337114254904 * 2))
        self.assertTrue(np.isclose((func / 2).energy, 1.6821337114254904 / 2))

    def test_wt(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        rho_r = io.read_density(dftpy_data_path + "/Al_fde_rho.pp")
        thefuncclass = Functional(type='KEDF', name='WT')
        func = thefuncclass(rho=rho_r)
        self.assertTrue(np.isclose(func.energy, 2.916818700014412))

    def test_lmgp(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        rho_r = io.read_density(dftpy_data_path + "/Al_fde_rho.pp")
        thefuncclass = Functional(type='KEDF', name='LMGP')
        func = thefuncclass(rho=rho_r)
        print(func.energy)
        self.assertTrue(np.isclose(func.energy, 2.9145872498710492, atol = 1E-3))


if __name__ == "__main__":
    unittest.main()
