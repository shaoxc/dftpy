from ase.calculators.interface import Calculator
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory
from ase import units
import ase.io

from dftpy.config.config import DefaultOption, PrintConf, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator
import pathlib
dftpy_data_path = pathlib.Path(__file__).resolve().parents[1] / 'DATA'

############################## initial config ##############################
conf = DefaultOption()
conf['PATH']['pppath'] = dftpy_data_path
conf['PP']['Al'] = 'al.lda.recpot'
conf['JOB']['calctype'] = 'Energy Force Stress'
conf['OPT']['method'] = 'TN'
conf['KEDF']['kedf'] = 'WT'
conf['OUTPUT']['stress'] = False
conf = OptionFormat(conf)
PrintConf(conf)
#-----------------------------------------------------------------------
path = conf['PATH']['pppath']
atoms = ase.io.read(path+'/'+'fcc.vasp')
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

opt.run(fmax = 0.05)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)
