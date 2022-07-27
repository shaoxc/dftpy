#!/usr/bin/env python3
import os
import unittest
import numpy as np

# from dftpy.config.config import DefaultOption, ConfSpecialFormat
from dftpy.config import DefaultOption, OptionFormat
from dftpy.interface import OptimizeDensityConf
from dftpy.api.api4ase import DFTpyCalculator
import pytest


class Test(unittest.TestCase):
    def test_md(self):
        pytest.importorskip("ase")
        from ase.lattice.cubic import FaceCenteredCubic
        from ase.md.langevin import Langevin
        from ase.md.verlet import VelocityVerlet
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.io.trajectory import Trajectory
        from ase import units
        from ase.io import write, read
        dftpy_data_path = os.environ.get('DFTPY_DATA_PATH')
        conf = DefaultOption()
        conf['PATH']['pppath'] = os.environ.get('DFTPY_DATA_PATH')
        conf['PP']['Al'] = '/Al_lda.oe01.recpot'
        conf['OPT']['method'] = 'TN'
        conf['KEDF']['kedf'] = 'WT'
        conf['EXC']['xc'] = 'LDA'
        conf['JOB']['calctype'] = 'Energy Force'
        conf['OUTPUT']['time'] = False
        conf = OptionFormat(conf)
        calc = DFTpyCalculator(config=conf)
        atoms = read(filename=dftpy_data_path + '/initial_atoms_md.traj',
                     format='traj',
                     index=-1)
        atoms.calc = calc
        dyn = VelocityVerlet(atoms, 2 * units.fs)
        dyn.run(3)
        atoms_fin = read(filename=dftpy_data_path + '/md.traj', index=-1)
        print(atoms.positions - atoms_fin.positions)
        self.assertTrue(np.isclose(atoms.get_momenta(), atoms_fin.get_momenta(), atol=1.e-3).all())


if __name__ == "__main__":
    unittest.main()
