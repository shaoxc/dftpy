#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.formats.io import read
from dftpy.ewald import ewald
from dftpy.functional.pseudo import LocalPseudo


class Test(unittest.TestCase):
    def setUp(self):
        self.dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        self.mol= read(self.dftpy_data_path + "/Al_fde_rho.pp", kind=all)
        self.ref_energy = 1.3497046

    def test_recpot(self):
        PP_list = {'Al': self.dftpy_data_path + "/al.lda.recpot"}
        self._run_energy(PP_list)

    def test_upf(self):
        PP_list = {'Al': self.dftpy_data_path + "/al.lda.upf"}
        self._run_energy(PP_list)

    def test_psp(self):
        PP_list = {'Al': self.dftpy_data_path + "/al.lda.psp"}
        self._run_energy(PP_list)

    def _run_energy(self, PP_list):
        PSEUDO = LocalPseudo(grid=self.mol.cell, ions=self.mol.ions, PP_list=PP_list)
        energy = PSEUDO(self.mol.field).energy

        print('energy', energy, self.ref_energy)
        self.assertTrue(np.isclose(energy, self.ref_energy, atol=1.E-4))


if __name__ == "__main__":
    unittest.main()
