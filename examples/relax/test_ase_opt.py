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

from dftpy.config.config import DefaultOption, ConfSpecialFormat, PrintConf
from dftpy.interface import OptimizeDensityConf
from dftpy.api.api4ase import DFTpyCalculator

############################## initial config ##############################
conf = DefaultOption()
conf['PATH']['pppath'] = os.environ.get('DFTPY_DATA_PATH') 
# conf['PP']['Ga'] = 'Ga_lda.oe04.recpot'
# conf['PP']['As'] = 'As_lda.oe04.recpot'
# conf['PP']['Al'] = 'Al_lda.oe01.recpot'
conf['PP']['Al'] = 'al.lda.recpot'
conf['JOB']['calctype'] = 'Energy Force Stress'
# conf['JOB']['calctype'] = 'Energy Force'
conf['OPT']['method'] = 'TN'
# conf['KEDF']['kedf'] = 'x_TF_y_vW'
# conf['OUTPUT']['time'] = False
conf['OUTPUT']['stress'] = False
# conf['MATH']['reuse'] = False
conf = ConfSpecialFormat(conf)
PrintConf(conf)
#-----------------------------------------------------------------------
path = os.environ.get('DFTPY_DATA_PATH') 
# atoms = ase.io.read(path+'/'+'GaAs.vasp')
atoms = ase.io.read(path+'/'+'fcc.vasp')
# atoms = ase.io.read(path+'/'+'20.vasp')
trajfile = 'opt.traj'

calc = DFTpyCalculator(config = conf)
atoms.set_calculator(calc)

############################## Relaxation type ##############################
'''
Ref : 
    https ://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize
    https ://wiki.fysik.dtu.dk/ase/ase/constraints.html
'''
af = atoms
af = StrainFilter(atoms)
# af = UnitCellFilter(atoms)
############################## Relaxation method ##############################
# opt = BFGS(af, trajectory = trajfile)
# opt = LBFGS(af, trajectory = trajfile, memory = 10, use_line_search = True)
# opt = LBFGS(af, trajectory = trajfile, memory = 10, use_line_search = False)
opt = SciPyFminCG(af, trajectory = trajfile)
# opt = SciPyFminBFGS(af, trajectory = trajfile)

opt.run(fmax = 0.001)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)
