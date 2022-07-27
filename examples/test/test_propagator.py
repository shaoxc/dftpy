#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.functional.pseudo import LocalPseudo
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.utils.utils import calc_rho
from dftpy.formats import io
from dftpy.constants import Units
from dftpy.td.utils import initial_kick

ang2bohr = 1.0/Units.Bohr


class TestPropagator(unittest.TestCase):

    def setUp(self):
        self.hamiltonian = Hamiltonian()
        self.interval = 1e-3
        self.taylor = Propagator(self.hamiltonian, self.interval, name='taylor',
                                 order=5)
        self.cn = Propagator(self.hamiltonian, self.interval, name='crank-nicholson')

    def test_call(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        ions, rho0, _ = io.read_all(dftpy_data_path + '/GaAs_random.xsf', full=True)
        rho0 *= ang2bohr ** 3 # why?

        KE = Functional(type='KEDF', name='TF')
        XC = Functional(type='XC', name='LDA')
        HARTREE = Functional(type="HARTREE")
        PPlist = {'Ga': dftpy_data_path + '/Ga_lda.oe04.recpot',
                  'As': dftpy_data_path + '/As_lda.oe04.recpot'}
        PSEUDO = LocalPseudo(grid=rho0.grid, ions=ions, PP_list=PPlist)
        E_v_Evaluator = TotalFunctional(
            KineticEnergyFunctional=KE,
            XCFunctional=XC,
            HARTREE=HARTREE,
            PSEUDO=PSEUDO)

        x = 0
        k = 1.0e-3
        psi0 = initial_kick(k, x, np.sqrt(rho0))

        psi = psi0
        func = E_v_Evaluator.compute(rho0, calcType=["V"])
        self.hamiltonian.v = func.potential
        E0 = self.hamiltonian.energy(psi)

        for i_t in range(10):
            psi, info = self.taylor(psi)
            rho = calc_rho(psi)
            func = E_v_Evaluator.compute(rho, calcType=["V"])
            self.hamiltonian.v = func.potential

        E = self.hamiltonian.energy(psi)
        self.assertTrue(np.isclose(E, E0, rtol=1e-3))

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        print(delta_mu[0])
        self.assertTrue(np.isclose(delta_mu[0], 1.1458e-02, rtol=1e-3))

        psi = psi0
        func = E_v_Evaluator.compute(rho0, calcType=["V"])
        self.hamiltonian.v = func.potential
        for i_t in range(10):
            psi, info = self.cn(psi)
            rho = calc_rho(psi)
            func = E_v_Evaluator.compute(rho, calcType=["V"])
            self.hamiltonian.v = func.potential

        E = self.hamiltonian.energy(psi)
        self.assertTrue(np.isclose(E, E0, rtol=1e-3))

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        self.assertTrue(np.isclose(delta_mu[0], 1.1458e-02, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
