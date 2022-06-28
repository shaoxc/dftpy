from dftpy.optimization import Optimization
from dftpy.functional import Functional, LocalPseudo, TotalFunctional
from dftpy.formats import io
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import ecut2nr
from dftpy.mpi import MP, sprint

#
mp = MP()
# mp = MP(parallel = True)
#
ions = io.read('../DATA/fcc.vasp')
#
# PP_list = {'Al': '../DATA/al.lda.recpot'}
PP_list = ['../DATA/al.lda.recpot']
#
nr = ecut2nr(ecut = 20, lattice = ions.cell)
grid = DirectGrid(lattice=ions.cell, nr=nr, mp=mp)
rho_ini = DirectField(grid=grid)
#
PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list)
KE = Functional(type='KEDF', name='WT')
XC = Functional(type='XC', name='LDA')
HARTREE = Functional(type='HARTREE')
#
rho_ini[:] = ions.get_ncharges() / ions.cell.volume
#
funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
EnergyEvaluator = TotalFunctional(**funcDict)
#
opt = Optimization(EnergyEvaluator=EnergyEvaluator)
#
new_rho = opt.optimize_rho(guess_rho=rho_ini)
#
sprint('Calc Energy', comm = mp)
Enew = EnergyEvaluator.Energy(rho=new_rho, ions=ions)
sprint('Energy New (a.u.)', Enew, comm = mp)
