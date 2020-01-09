#!/usr/bin/env python3
import os
import unittest
import numpy as np
import dftpy.formats.io as dftpy_io
from dftpy.optimization import Optimization
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import bestFFTsize
from dftpy.pseudo import LocalPseudo


class Test(unittest.TestCase):
    def test_optim(self):
        path_pp = os.environ.get("DFTPY_DATA_PATH") + "/"
        path_pos = os.environ.get("DFTPY_DATA_PATH") + "/"
        file1 = "Ga_lda.oe04.recpot"
        file2 = "As_lda.oe04.recpot"
        posfile = "GaAs_random.vasp"
        ions = dftpy_io.read(path_pos + posfile, names=["Al"])
        lattice = ions.pos.cell.lattice
        metric = np.dot(lattice.T, lattice)
        gap = 0.4
        nr = np.zeros(3, dtype="int32")
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i]) / gap)
        print("The initial grid size is ", nr)
        for i in range(3):
            nr[i] = bestFFTsize(nr[i])
        print("The final grid size is ", nr)
        grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=False)
        zerosA = np.zeros(grid.nnr, dtype=float)
        rho_ini = DirectField(grid=grid, griddata_F=zerosA, rank=1)
        PP_list = {"Ga": path_pp + file1, "As": path_pp + file2}
        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=True)
        optional_kwargs = {}
        KE = FunctionalClass(type="KEDF", name="WT", optional_kwargs=optional_kwargs)
        XC = FunctionalClass(type="XC", name="LDA")
        HARTREE = FunctionalClass(type="HARTREE")

        charge_total = 0.0
        for i in range(ions.nat):
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini[:] = charge_total / ions.pos.cell.volume
        E_v_Evaluator = TotalEnergyAndPotential(
            KineticEnergyFunctional=KE, XCFunctional=XC, HARTREE=HARTREE, PSEUDO=PSEUDO,
        )
        optimization_options = {
            "econv": 1e-6,  # Energy Convergence (a.u./atom)
            "maxfun": 50,  # For TN method, it's the max steps for searching direction
            "maxiter": 100,  # The max steps for optimization
        }
        optimization_options["econv"] *= ions.nat
        opt = Optimization(
            EnergyEvaluator=E_v_Evaluator, optimization_options=optimization_options, optimization_method="TN",
        )
        new_rho = opt.optimize_rho(guess_rho=rho_ini)
        Enew = E_v_Evaluator.Energy(rho=new_rho, ions=ions, usePME=True)

        self.assertTrue(np.isclose(Enew, -43.3046))


if __name__ == "__main__":
    unittest.main()
