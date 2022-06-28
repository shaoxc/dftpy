#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.formats import io
from dftpy.ewald import ewald
from dftpy.functional.pseudo import LocalPseudo


class Test(unittest.TestCase):
    def test_ewald_PME(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        print()
        print("*" * 50)
        print("Testing particle mesh Ewald method")
        ions, rho, _ = io.read_all(dftpy_data_path + "/Al_fde_rho.pp")
        Ewald_ = ewald(rho=rho, ions=ions, verbose=False)
        Ewald_PME = ewald(rho=rho, ions=ions, verbose=False, PME=True)

        e_corr = Ewald_.Energy_corr()
        e_real = Ewald_.Energy_real()
        e_rec = Ewald_.Energy_rec()
        e_real_pme = Ewald_PME.Energy_real_fast2()
        e_rec_pme = Ewald_PME.Energy_rec_PME()

        f_real = Ewald_.Forces_real()
        f_rec = Ewald_.Forces_rec()
        f_rec_pme = Ewald_PME.Forces_rec_PME()

        s_real = Ewald_.Stress_real()
        s_rec = Ewald_.Stress_rec()
        s_rec_pme = Ewald_PME.Stress_rec_PME()

        self.assertTrue(np.isclose(e_corr, -13.003839613564997, atol=1.E-5))
        self.assertTrue(np.isclose(e_real, 0.0, atol=1.E-5))
        self.assertTrue(np.isclose(e_rec, 7.606943756474895, atol=1.E-5))
        self.assertTrue(np.isclose(e_real_pme, 0.0, atol=1.E-5))
        self.assertTrue(np.isclose(e_rec_pme, 7.606943756474895, atol=1.E-5))
        self.assertTrue(np.isclose(f_real[0, 0], 0.0, atol=1.E-5))
        self.assertTrue(np.isclose(f_rec[0, 0], 0.0, atol=1.E-5))
        self.assertTrue(np.isclose(f_rec_pme[0, 0], 0.0, atol=1.E-5))
        self.assertTrue(np.isclose(s_real[0, 0], 0.0, atol=1.E-5))
        self.assertTrue(np.isclose(s_rec[0, 0], 8.04967853e-03, atol=1.E-5))
        self.assertTrue(np.isclose(s_rec_pme[0, 0], 8.04967853e-03, atol=1.E-5))

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
