#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functionals import FunctionalClass
from dftpy.constants import LEN_CONV
from dftpy.formats.qepp import PP


class Test(unittest.TestCase):
    def test_gga(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        rho_r = mol.field
        optional_kwargs_gga = {}
        optional_kwargs_gga['k_str'] = 'lc94'
        thefuncclass = FunctionalClass(type='KEDF',
                                       name='GGA',
                                       **optional_kwargs_gga)
        func = thefuncclass(rho=rho_r)
        self.assertTrue(np.isclose(func.energy, 1.6821337114254904))
        self.assertTrue(np.isclose((func + func).energy, 1.6821337114254904 * 2))
        self.assertTrue(np.isclose((func * 2).energy, 1.6821337114254904 * 2))
        self.assertTrue(np.isclose((func / 2).energy, 1.6821337114254904 / 2))

    def test_wt(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        rho_r = mol.field
        thefuncclass = FunctionalClass(type='KEDF', name='WT')
        func = thefuncclass(rho=rho_r)
        self.assertTrue(np.isclose(func.energy, 2.916818700014412))

    def test_lmgp(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        rho_r = mol.field
        thefuncclass = FunctionalClass(type='KEDF', name='LMGP')
        func = thefuncclass(rho=rho_r)
        print(func.energy)
        # self.assertTrue(np.isclose(func.energy, 2.9146409624966725)) # with np.geomspace numpy.1.16
        self.assertTrue(np.isclose(func.energy, 2.9147420609863923))


if __name__ == "__main__":
    unittest.main()
