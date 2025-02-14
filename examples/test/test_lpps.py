import unittest
import numpy as np
import dftpy.formats.io as dftpy_io
from dftpy.optimization import Optimization
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import ecut2nr
from dftpy.functional.pseudo import LocalPseudo
from common import dftpy_data_path

class Test(unittest.TestCase):
    def test_lpps(self):
        from dftpy.ions import Ions
        from dftpy.field import DirectField
        from dftpy.grid import DirectGrid
        from dftpy.functional import LocalPseudo, Functional, TotalFunctional
        from dftpy.formats import io
        from dftpy.optimization import Optimization
        from dftpy.mpi import sprint
        from dftpy.functional.pseudo.psp import PSP
        from dftpy.constants import environ
        from scipy.optimize import minimize
        ions, rho_target, _ = io.read_all(dftpy_data_path / 'rho_target.xsf')
        grid = rho_target.grid
        PP_list = {'Au': dftpy_data_path / 'au_lda_v1.uspp.F.UPF'}
        MaxPoints=1000 
        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, MaxPoints=MaxPoints)
        rho_ini = rho_target.copy()
        KE = Functional(type='KEDF',name='TFvW', y=0.2)
        XC = Functional(type='XC',name='LDA', libxc=False)
        HARTREE = Functional(type='HARTREE')
        evaluator = TotalFunctional(KE=KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
        optimization_options = {'econv' : 1e-6*ions.nat}
        opt = Optimization(EnergyEvaluator=evaluator, optimization_options = optimization_options, optimization_method = 'TN')
        rho = opt.optimize_rho(guess_rho=rho_ini)
        def delta_pp(r, rcut, a):
            d = r - rcut
            b = (3*a[0]*rcut-4*a[1]*rcut**2+5*a[2]*rcut**3)/2.0
            v = b*d**2 + a[0]*d**3 + a[1]*d**4+a[2]*d**5
            v[r>rcut] = 0.0
            return v
        def lpp2vloc(r, v, ions, grid, zval=0.0):
            engine = PSP(None)
            engine.r = r
            engine.v = v
            engine._zval = zval
            pseudo = LocalPseudo(grid = grid, ions=ions, PP_list={'Au':engine}, MaxPoints=MaxPoints)
            pseudo.local_PP()
            return pseudo._vreal

        grid = rho_target.grid
        rcut = 2.35
        r = np.linspace(0, rcut, 100)
        a = np.zeros(3)
        
        ext = Functional(type='EXT')
        evaluator.UpdateFunctional(newFuncDict={'EXT': ext})
        
        opt = Optimization(EnergyEvaluator=evaluator)
        
        rho_ini = rho_target.copy()
        environ['LOGLEVEL'] = 4
        def delta_rho(a):
            v = delta_pp(r, rcut, a)
            ext.v = lpp2vloc(r, v, ions, grid)
            rho = opt.optimize_rho(guess_rho=rho_ini)
            rho_ini[:]=rho
            diff = 0.5 * (np.abs(rho - rho_target)).integral()
            return diff
        
        res = minimize(delta_rho, a, method='Powell', options={'ftol': 1.0e-4})

        self.assertTrue(np.isclose(res.fun, 0.1195192, atol=1.e-3))

if __name__ == "__main__":
    unittest.main()
