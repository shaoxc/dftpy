#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functionals import FunctionalClass
from dftpy.constants import LEN_CONV
from dftpy.semilocal_xc import XC, PBE
from dftpy.formats.qepp import PP


class Test(unittest.TestCase):
    def test_gga(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        rho_r = mol.field
        optional_kwargs_gga = {}
        optional_kwargs_gga['k_str'] = 'lc94'
        optional_kwargs_gga['polarization'] = 'unpolarized'
        thefuncclass = FunctionalClass(type='KEDF',
                                       name='GGA',
                                       optional_kwargs=optional_kwargs_gga)
        func = thefuncclass(rho=rho_r)
        self.assertTrue(np.isclose(func.energy, 1.6821337114254904))

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
        self.assertTrue(np.isclose(func.energy, 2.9146409624966725))


if __name__ == "__main__":
    unittest.main()
