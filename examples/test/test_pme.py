#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functionals import FunctionalClass
from dftpy.constants import LEN_CONV
from dftpy.formats.qepp import PP
from dftpy.ewald import ewald
from dftpy.pseudo import LocalPseudo


class Test(unittest.TestCase):
    def test_ie(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*" * 50)
        print("Testing loading pseudopotentials")
        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        PP_list = {'Al': dftpy_data_path + "/Al_lda.oe01.recpot"}
        ions = mol.ions
        grid = mol.cell
        rho = mol.field

        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=False)
        func = PSEUDO(density=rho)
        a = func.potential
        IE_Energy = func.energy
        IE_Force = PSEUDO.force(rho)
        #print(grid.nr, grid.dV, grid.lattice, grid.origin, grid.full)
        #print(grid.r)
        IE_Stress = PSEUDO.stress(rho)
        #print(grid.nr, grid.dV, grid.lattice, grid.origin, grid.full)
        #print(grid.r)

        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        grid=mol.cell
        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=True)
        func = PSEUDO(density=rho)
        IE_Energy_PME = func.energy
        IE_Force_PME = PSEUDO.force(rho)
        IE_Stress_PME = PSEUDO.stress(rho)

        print('IE energy', IE_Energy, IE_Energy_PME)
        self.assertTrue(np.isclose(IE_Energy, IE_Energy_PME, atol=1.E-4))
        print('IE forces', IE_Force, IE_Force_PME)
        self.assertTrue(np.allclose(IE_Force, IE_Force_PME, atol=1.E-4))
        print('IE stress', IE_Stress, IE_Stress_PME)
        self.assertTrue(np.allclose(IE_Stress, IE_Stress_PME, atol=1.E-4))
        #self.assertTrue(False)

    def test_ewald_PME(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*" * 50)
        print("Testing particle mesh Ewald method")
        mol = PP(filepp=dftpy_data_path + "/Al_fde_rho.pp").read()
        Ewald_ = ewald(rho=mol.field, ions=mol.ions, verbose=False)
        Ewald_PME = ewald(rho=mol.field,
                          ions=mol.ions,
                          verbose=False,
                          PME=True)

        print('Ewald energy', Ewald_.energy, Ewald_PME.energy)
        self.assertTrue(
            np.allclose(Ewald_.energy, Ewald_PME.energy, atol=1.E-5))
        print('Ewald forces', Ewald_.forces, Ewald_PME.forces)
        self.assertTrue(
            np.allclose(Ewald_.forces, Ewald_PME.forces, atol=1.E-5))
        print('Ewald stress', Ewald_.stress, Ewald_PME.stress)
        self.assertTrue(
            np.allclose(Ewald_.stress, Ewald_PME.stress, atol=1.E-5))


if __name__ == "__main__":
    unittest.main()
