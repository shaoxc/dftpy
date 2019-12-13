import os
import numpy as np
from ase.calculators.interface import Calculator
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory
from ase import units
import ase.io

from dftpy.config import DefaultOption, OptionFormat
from dftpy.interface import OptimizeDensityConf
from dftpy.api.api4ase import DFTpyCalculator

############################## initial config ##############################
conf = DefaultOption()
conf['PATH']['pppath'] = os.environ.get('DFTPY_DATA_PATH') 
conf['PP']['Al'] = 'Al_lda.oe01.recpot'
conf['JOB']['calctype'] = 'Energy Force Stress'
conf['KEDF']['kedf'] = 'WT'
conf = OptionFormat(conf)
#-----------------------------------------------------------------------
path = os.environ.get('DFTPY_DATA_PATH') 
atoms = ase.io.read(path+'/'+'fcc.vasp')

calc = DFTpyCalculator(config = conf)
atoms.set_calculator(calc)

trajfile = 'opt.traj'

af = atoms
af = StrainFilter(atoms)
af = UnitCellFilter(atoms)

opt = SciPyFminCG(af, trajectory = trajfile)

opt.run(fmax = 0.01)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)
