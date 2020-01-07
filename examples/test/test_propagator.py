import unittest
import numpy as np

from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.pseudo import LocalPseudo
from dftpy.propagator import Propagator, hamiltonian
from dftpy.tdrunner import cal_rho_j
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.system import System
from dftpy.formats.xsf import read_xsf

class TestPropagator(unittest.TestCase):

    def setUp(self):
        self.taylor = Propagator(interval = 1.0e-3, type='taylor', optional_kwargs=\
            {"order":5})
        self.cn = Propagator(interval = 1.0e-3, type='crank-nicolson')

    def test_init(self):
        self.assertEqual(self.taylor.type, 'taylor')
        self.assertEqual(self.taylor.interval, 1.0e-3)
        self.assertEqual(self.taylor.optional_kwargs['order'], 5)
        self.assertEqual(self.cn.type, 'crank-nicolson')
        self.assertEqual(self.cn.interval, 1.0e-3)

    def test_call(self):
        sys = read_xsf('../DATA/GaAs_random.xsf',full=True)
        
        KE = FunctionalClass(type='KEDF', name='TF')
        XC = FunctionalClass(type='XC', name='LDA')
        HARTREE = FunctionalClass(type="HARTREE")
        PPlist = {'Ga':'../DATA/Ga_lda.oe04.recpot',
            'As':'../DATA/As_lda.oe04.recpot'}
        PSEUDO = LocalPseudo(grid=sys.cell, ions=sys.ions, PP_list=PPlist)
        E_v_Evaluator = TotalEnergyAndPotential(
            KineticEnergyFunctional=KE, 
            XCFunctional=XC, 
            HARTREE=HARTREE,
            PSEUDO=PSEUDO)

        rho0 = sys.field
        x = rho0.grid.r[0]
        k = 1.0e-6
        psi0 = np.sqrt(rho0) * np.exp(1j * k * x)
        psi0.cplx = True

        psi = psi0
        func = E_v_Evaluator.ComputeEnergyPotential(rho0, calcType="Potential")
        potential = func.potential
        E0 = np.real(np.conj(psi) * hamiltonian(psi, potential)).integral()
        for i_t in range(10):
            psi, info = self.taylor(psi, potential)
            rho, _ = cal_rho_j(psi)
            func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType="Potential")
            potential = func.potential

        E = np.real(np.conj(psi) * hamiltonian(psi, potential)).integral()
        self.assertTrue(np.isclose(E, E0, rtol=1e-3))
        
        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        self.assertTrue(np.isclose(delta_mu[0], -1.3517e-05, rtol=1e-3))

        psi = psi0
        func = E_v_Evaluator.ComputeEnergyPotential(rho0, calcType="Potential")
        potential = func.potential
        for i_t in range(10):
            psi, info = self.cn(psi, potential)
            rho, _ = cal_rho_j(psi)
            func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType="Potential")
            potential = func.potential

        E = np.real(np.conj(psi) * hamiltonian(psi, potential)).integral()
        self.assertTrue(np.isclose(E, E0, rtol=1e-3))
        
        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        self.assertTrue(np.isclose(delta_mu[0], -1.3517e-05, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
