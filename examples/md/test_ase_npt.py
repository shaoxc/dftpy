import os
import numpy as np
from ase.calculators.interface import Calculator
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from dftpy.config import DefaultOption, OptionFormat
from dftpy.interface import OptimizeDensityConf
from dftpy.api.api4ase import DFTpyCalculator

############################## initial config ##############################
conf = DefaultOption()
conf['PATH']['pppath'] = os.environ.get('DFTPY_DATA_PATH') 
conf['PATH']['pppath'] = os.environ.get('DFTPY_DATA_PATH') 
# conf['PP']['Al'] = 'Al_lda.oe01.recpot'
conf['PP']['Al'] = 'al.lda.recpot'
conf['JOB']['calctype'] = 'Energy Force Stress'
conf['OPT']['method'] = 'TN'
# conf['KEDF']['kedf'] = 'x_TF_y_vW'
conf['OUTPUT']['time'] = 'False'
conf['OUTPUT']['stress'] = 'False'
conf = OptionFormat(conf)
print(conf)
#-----------------------------------------------------------------------
'''
Ref :
    https ://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md
    https ://wiki.fysik.dtu.dk/ase/_modules/ase/md/npt.html#NPT
'''

size = 3
a = 4.24068463425528
T = 1023  # Kelvin
T *= units.kB
atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          latticeconstant = a,
                          symbol="Al",
                          size=(size, size, size),
                          pbc=True)

calc = DFTpyCalculator(config = conf)
atoms.set_calculator(calc)

externalstress = 0.0
ttime = 10 * units.fs
# pfactor = 75 **2 * B
pfactor = 0.6

dyn = NPT(atoms, 2 * units.fs, T, externalstress, ttime, pfactor)

step = 0
interval = 1
def printenergy(a=atoms):  
    global step, interval
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    spot = a.get_stress()
    press = np.sum(spot[:3])/3.0
    print('Step={:<d} Epot={:.5f} Ekin={:.5f} T={:.3f} Etot={:.5f} Press={:.5f}'.format(\
            step, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, press/units.GPa))
    step += interval
    
dyn.attach(printenergy, interval=1)

traj = Trajectory('npt.traj', 'w', atoms)
dyn.attach(traj.write, interval=1)
# gibbs=a.get_gibbs_free_energy()
dyn.run(500)
