#!/usr/bin/env python3
import os
import unittest
import numpy as np
import pytest

from dftpy.config.config import DefaultOption, ConfSpecialFormat
from dftpy.interface import OptimizeDensityConf
from dftpy.api.api4ase import DFTpyCalculator

class Test(unittest.TestCase):
    def test_opt(self):
        pytest.importorskip("ase")
        from ase.calculators.interface import Calculator
        from ase.lattice.cubic import FaceCenteredCubic
        from ase.optimize import BFGS, LBFGS, FIRE
        from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
        from ase.constraints import StrainFilter, UnitCellFilter
        from ase.io.trajectory import Trajectory
        from ase import units
        from ase.io import read, write
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        conf = DefaultOption()
        conf['PATH']['pppath'] = os.environ.get('DFTPY_DATA_PATH')
        conf['PP']['Al'] = 'al.lda.recpot'
        conf['JOB']['calctype'] = 'Energy Force Stress'
        conf['OPT']['method'] = 'TN'
        conf['OUTPUT']['stress'] = False
        conf['EXC']['xc'] = 'LDA'
        conf = ConfSpecialFormat(conf)
        path = os.environ.get('DFTPY_DATA_PATH')
        atoms = read(path+'/'+'fcc.vasp')
        calc = DFTpyCalculator(config = conf)
        atoms.calc = calc
        af = StrainFilter(atoms)
        opt = SciPyFminCG(af)
        opt.run(fmax = 0.001)
        atoms_final = read(filename=dftpy_data_path+'/ase_opt.traj',format='traj')
        self.assertTrue(np.isclose(atoms.positions, atoms_final.positions, atol=1.e-3).all())

if __name__ == "__main__":
    unittest.main()
