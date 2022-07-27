#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.formats import io
from dftpy.ewald import ewald
from dftpy.functional.pseudo import LocalPseudo


class Test(unittest.TestCase):
    def test_ie(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*" * 50)
        print("Testing loading pseudopotentials")
        ions, rho, _ = io.read_all(dftpy_data_path + "/Al_fde_rho.pp")
        PP_list = {'Al': dftpy_data_path + "/Al_lda.oe01.recpot"}
        grid = rho.grid

        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=False)
        func = PSEUDO(rho)
        IE_Energy = func.energy
        IE_Force = PSEUDO.force(rho)
        IE_Stress = PSEUDO.stress(rho, energy=IE_Energy)

        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=True)
        func = PSEUDO(rho)
        IE_Energy_PME = func.energy
        IE_Force_PME = PSEUDO.force(rho)
        IE_Stress_PME = PSEUDO.stress(rho, energy=IE_Energy_PME)

        print('IE energy', IE_Energy, IE_Energy_PME)
        self.assertTrue(np.isclose(IE_Energy, IE_Energy_PME, atol=1.E-4))
        print('IE forces', IE_Force, IE_Force_PME)
        self.assertTrue(np.allclose(IE_Force, IE_Force_PME, atol=1.E-4))
        print('IE stress', IE_Stress, IE_Stress_PME)
        self.assertTrue(np.allclose(IE_Stress, IE_Stress_PME, atol=1.E-4))

    def test_ewald_PME(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*" * 50)
        print("Testing particle mesh Ewald method")
        ions, rho, _ = io.read_all(dftpy_data_path + "/Al_fde_rho.pp")
        Ewald_ = ewald(rho=rho, ions=ions, verbose=False)
        Ewald_PME = ewald(rho=rho, ions=ions, verbose=False, PME=True)

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
