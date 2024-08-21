import os
import numpy as np
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.optimization import Optimization
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.formats.vasp import read_POSCAR
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.utils.utils import calc_rho, calc_j
from dftpy.td.utils import initial_kick
from dftpy.mpi import pmi, sprint, mp
from dftpy.ions import Ions
#-----------------------------------------------------------------------
if pmi.size > 0:
    from mpi4py import MPI
    mp.comm = MPI.COMM_WORLD
#-----------------------------------------------------------------------
import ase
#drom ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.calculators.mixing import LinearCombinationCalculator, MixedCalculator, SumCalculator
from ase import units,Atoms

from ase.build import bulk, molecule
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.calculators.eam import EAM

from dftpy.config.config import DefaultOption, OptionFormat, PrintConf
from dftpy.api.api4ase import DFTpyCalculator


import pathlib
dftpy_data_path = pathlib.Path(__file__).resolve().parents[1] / 'DATA'
np.random.seed(8888)


############################## initial config ##############################
"""
Ref :
    https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md
    https://wiki.fysik.dtu.dk/ase/_modules/ase/md/npt.html#NPT
"""
size = 1 
a = 4.24068463425528 
T = 600.0  # Kelvin
DATA='../DATA/'
#atoms = Atoms('Al', positions=[(a/2,a/2,a/2)], pbc=False, cell=[[a, 0, 0], [0, a, 0], [0, 0, a]])
atoms =  bulk('Al', 'fcc', a=a, cubic=True)

ions = Ions.from_ase(atoms)

# ------------------DFT BO-----------------------------------------------------
#
path = str(dftpy_data_path)
PP_list = {'Al': path+os.sep+'al.lda.recpot'}
grid = DirectGrid(ecut=20, lattice=ions.cell, mp=mp)
rho = DirectField(grid=grid)

PSEUDO = Functional(type='PSEUDO', grid=grid, ions=ions, PP_list=PP_list)
KE = Functional(type='KEDF', name='TFvW')
XC = Functional(type='XC', name='LDA')
HARTREE = Functional(type='HARTREE')
#
funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
EnergyEvaluator = TotalFunctional(**funcDict)
opt = {'econv': 1e-12,'maxiter': 500}
#
rho[:] = ions.get_ncharges() / ions.cell.volume
optimizer = Optimization(EnergyEvaluator=EnergyEvaluator, optimization_method='TN',optimization_options=opt )
calcBO = DFTpyCalculator(optimizer = optimizer, evaluator = EnergyEvaluator, rho = rho)

# --------------TDDFT EF---------------------------------------------------------


nas = 1

conf = DefaultOption()
conf["PATH"]["pppath"] = dftpy_data_path
conf["PP"]["Al"] = "al.lda.recpot"
conf["OPT"]["method"] = "TN"
conf["OPT"]["econv"] = 1e-12
conf["KEDF"]["kedf"] = "TFvW"
conf["JOB"]["calctype"] = "Energy Force"
conf["OUTPUT"]["time"] = False
conf["TD"]["single_step"] = True
conf["TD"]["max_pc"] = 2
conf["TD"]["timestep"] = 0.041341374575751*nas
conf["TD"]["strength"] = 0.00 
conf["GRID"]["cplx"] = True
conf["GRID"]["gfull"] = True
conf["GRID"]["ecut"] =  540
#conf["GRID"]["nr"] = "16 16 16"
conf["PROPAGATOR"]["propagator"] = "crank-nicholson"
conf["PROPAGATOR"]["tol"] = 1e-14
conf["PROPAGATOR"]["atol"] = 1e-14

conf = OptionFormat(conf)
PrintConf(conf)

calcEF = DFTpyCalculator(rho=rho,config=conf,optimizer = optimizer, evaluator = EnergyEvaluator, mp =mp, step = nas)

# ----------Force Field-------------------------------------------------------------
calcFF =  EAM(potential='Al_zhou.eam.alloy')
#-----mixing-----
calcA = [calcFF,calcBO,calcEF]
weights = [1.0,-1.0,1.0]

calcT= LinearCombinationCalculator(calcA,weights) 

atoms.calc = calcT
#-----dynamics---------
MaxwellBoltzmannDistribution(atoms, temperature_K = T, force_temp=True)

#dyn = Langevin(atoms, 0.001 * units.fs, temperature_K = T,  friction = 0.1)
dyn = VelocityVerlet(atoms, .001 * nas * units.fs )

step = 0
interval = 1
def printenergy(a=atoms):
    global step, interval
    epot = a.get_potential_energy() 
    ekin = a.get_kinetic_energy() 
    sprint(
            "Step={:<8d} Epot={:.5f} Ekin={:.5f} T={:.3f} Etot={:.5f} ".format(
            step, epot, ekin, ekin / (1.5 * units.kB), epot + ekin
        )
    )
    sprint(a.get_forces())
    step += interval

def check_stop():
    if os.path.isfile('dftpy_stopfile'): exit()


traj = Trajectory("testing.traj", "w", atoms)

dyn.attach(check_stop, interval=1)
dyn.attach(printenergy, interval=1)
dyn.attach(traj.write, interval=1)

dyn.run(10000)
