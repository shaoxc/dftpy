#!/usr/bin/env python3
import unittest
import numpy as np

from ase.build import bulk
from dftpy.ions import Ions
from dftpy.grid import DirectGrid
from dftpy.functional.pseudo import LocalPseudo
from dftpy.density import DensityGenerator
from common import dftpy_data_path

class Test(unittest.TestCase):
    def setUp(self):
        PP_list = {'Al': dftpy_data_path / "al.lda.upf"}
        self.ions = Ions.from_ase(bulk('Al', 'fcc', a=4.05*4, cubic=True))
        self.grid = DirectGrid(self.ions.cell, ecut=40)
        self.pseudo = LocalPseudo(grid=self.grid, ions=self.ions, PP_list=PP_list, MaxPoints=1000)

    def test_recipe(self):
        dg = DensityGenerator(pseudo=self.pseudo, direct=False)
        rho = dg.guess_rho(self.ions, grid=self.grid, ncharge=0.0)
        print('recipe', rho.integral())
        assert np.isclose(rho.integral(), 12.0, rtol=1E-5)

    def test_direct(self):
        dg = DensityGenerator(pseudo=self.pseudo, direct=True)
        rho = dg.guess_rho(self.ions, grid=self.grid, ncharge=0.0, rcut=10.0)
        print('direct', rho.integral())
        assert np.isclose(rho.integral(), 12.0, rtol=1E-3)


if __name__ == "__main__":
    unittest.main()
