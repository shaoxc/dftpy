from dftpy.optimization import Optimization
from dftpy.functional import Functional, LocalPseudo, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.mpi import MP, sprint
from dftpy.ions import Ions
from ase.build import bulk

mp = MP(parallel = False)
ions = Ions.from_ase(bulk('Al', 'fcc', a=4.05, cubic=True))
PP_list = {'Al': '../DATA/al.lda.recpot'}
grid = DirectGrid(ecut=20, lattice=ions.cell, mp=mp)
rho_ini = DirectField(grid=grid)
#
PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list)
KE = Functional(type='KEDF', name='WT')
XC = Functional(type='XC', name='LDA')
HARTREE = Functional(type='HARTREE')
#
funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
EnergyEvaluator = TotalFunctional(**funcDict)
#
rho_ini[:] = ions.get_ncharges() / ions.cell.volume
opt = Optimization(EnergyEvaluator=EnergyEvaluator)
new_rho = opt.optimize_rho(guess_rho=rho_ini)
#
Enew = EnergyEvaluator.Energy(rho=new_rho)
sprint('Energy New (a.u.)', Enew, comm = mp)
