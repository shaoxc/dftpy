#!/usr/bin/env python3
import os
import unittest
import numpy as np

from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.functional.pseudo import LocalPseudo
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.utils import calc_rho
from dftpy.formats.xsf import read_xsf
from dftpy.constants import LEN_CONV
ang2bohr = LEN_CONV["Angstrom"]["Bohr"]


class TestPropagator(unittest.TestCase):

    def setUp(self):
        self.hamiltonian = Hamiltonian()
        self.taylor = Propagator(self.hamiltonian, type='taylor',
            order = 5)
        self.cn = Propagator(self.hamiltonian, type='crank-nicolson')

    def test_init(self):
        self.assertEqual(self.taylor.type, 'taylor')
        self.assertEqual(self.taylor.optional_kwargs['order'], 5)
        self.assertEqual(self.cn.type, 'crank-nicolson')

    def test_call(self):
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        sys = read_xsf(dftpy_data_path+'/GaAs_random.xsf',full=True)
        sys.field *= ang2bohr ** 3

        KE = Functional(type='KEDF', name='TF')
        XC = Functional(type='XC', name='LDA')
        HARTREE = Functional(type="HARTREE")
        PPlist = {'Ga':dftpy_data_path+'/Ga_lda.oe04.recpot',
            'As':dftpy_data_path+'/As_lda.oe04.recpot'}
        PSEUDO = LocalPseudo(grid=sys.cell, ions=sys.ions, PP_list=PPlist)
        E_v_Evaluator = TotalFunctional(
            KineticEnergyFunctional=KE,
            XCFunctional=XC,
            HARTREE=HARTREE,
            PSEUDO=PSEUDO)

        rho0 = sys.field
        x = rho0.grid.r[0]
        k = 1.0e-3
        psi0 = np.sqrt(rho0) * np.exp(1j * k * x)
        psi0.cplx = True

        psi = psi0
        func = E_v_Evaluator.compute(rho0, calcType=["V"])
        self.hamiltonian.v = func.potential
        E0 = np.real(np.conj(psi) * self.hamiltonian(psi)).integral()
        int_t = 1e-3
        for i_t in range(10):
            psi, info = self.taylor(psi, int_t)
            rho = calc_rho(psi)
            func = E_v_Evaluator.compute(rho, calcType=["V"])
            self.hamiltonian.v = func.potential

        E = np.real(np.conj(psi) * self.hamiltonian(psi)).integral()
        self.assertTrue(np.isclose(E, E0, rtol=1e-3))

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        print(delta_mu[0])
        self.assertTrue(np.isclose(delta_mu[0], -1.1458e-02, rtol=1e-3))

        psi = psi0
        func = E_v_Evaluator.compute(rho0, calcType=["V"])
        self.hamiltonian.v = func.potential
        for i_t in range(10):
            psi, info = self.cn(psi, int_t)
            rho = calc_rho(psi)
            func = E_v_Evaluator.compute(rho, calcType=["V"])
            self.hamiltonian.v = func.potential

        E = np.real(np.conj(psi) * self.hamiltonian(psi)).integral()
        self.assertTrue(np.isclose(E, E0, rtol=1e-3))

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        self.assertTrue(np.isclose(delta_mu[0], 1.1458e-02, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
