#!/usr/bin/env python3
import os
import unittest
import numpy as np
import dftpy.formats.io as dftpy_io
from dftpy.optimization import Optimization
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import ecut2nr
from dftpy.time_data import TimeData
from dftpy.functional.pseudo import LocalPseudo


class Test(unittest.TestCase):
    def test_optim(self):
        path_pp = os.environ.get("DFTPY_DATA_PATH") + "/"
        path_pos = os.environ.get("DFTPY_DATA_PATH") + "/"
        file1 = "Ga_lda.oe04.recpot"
        file2 = "As_lda.oe04.recpot"
        posfile = "GaAs_random.vasp"
        ions = dftpy_io.read(path_pos + posfile, names=["Al"])
        nr = ecut2nr(lattice=ions.cell, spacing=0.4)
        print("The final grid size is ", nr)
        grid = DirectGrid(lattice=ions.cell, nr=nr, full=False)
        rho_ini = DirectField(grid=grid)
        PP_list = {"Ga": path_pp + file1, "As": path_pp + file2}
        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=True)
        optional_kwargs = {}
        KE = Functional(type="KEDF", name="WT", optional_kwargs=optional_kwargs)
        XC = Functional(type="XC", name="LDA")
        HARTREE = Functional(type="HARTREE")
        rho_ini[:] = ions.get_ncharges() / ions.cell.volume
        E_v_Evaluator = TotalFunctional(
            KineticEnergyFunctional=KE, XCFunctional=XC, HARTREE=HARTREE, PSEUDO=PSEUDO,
        )
        optimization_options = {
            "econv": 1e-6*ions.nat,
            "maxfun": 50,
            "maxiter": 100,
        }
        opt = Optimization(
            EnergyEvaluator=E_v_Evaluator, optimization_options=optimization_options, optimization_method="TN",
        )
        new_rho = opt.optimize_rho(guess_rho=rho_ini)
        Enew = E_v_Evaluator.Energy(rho=new_rho, ions=ions, usePME=True)

        self.assertTrue(np.isclose(Enew, -43.3046))


if __name__ == "__main__":
    unittest.main()
