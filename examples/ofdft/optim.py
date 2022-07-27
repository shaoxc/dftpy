from dftpy.optimization import Optimization
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.constants import ENERGY_CONV
from dftpy.formats import io
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import ecut2nr
from dftpy.time_data import TimeData
from dftpy.functional.pseudo import LocalPseudo

def test_optim():
    path_pp='../DATA/'
    path_pos='../DATA/'
    file1='al.lda.recpot'
    posfile='fcc.vasp'
    ions = io.read(path_pos+posfile, names=['Al'])
    nr = ecut2nr(lattice=ions.cell, spacing=0.4)
    print('The final grid size is ', nr)
    grid = DirectGrid(lattice=ions.cell, nr=nr, full=False)
    PP_list = {'Al': path_pp+file1}
    PSEUDO = LocalPseudo(grid = grid, ions=ions,PP_list=PP_list)
    optional_kwargs = {}
    KE = Functional(type='KEDF', name='WT', optional_kwargs=optional_kwargs)
    XC = Functional(type='XC', name='LDA')
    HARTREE = Functional(type='HARTREE')

    rho_ini = DirectField(grid=grid)
    rho_ini[:] = ions.get_ncharges() / ions.cell.volume
    funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
    E_v_Evaluator = TotalFunctional(**funcDict)
    optimization_options = {
            'econv' : 1e-6*ions.nat, # Energy Convergence
            'maxfun' : 50,  # For TN method, it's the max steps for searching direction
            'maxiter' : 100,# The max steps for optimization
            }
    opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_options = optimization_options,
            optimization_method = 'CG-HS')
            # optimization_method = 'TN')
    new_rho = opt.optimize_rho(guess_rho=rho_ini)
    print('Calc Energy')
    Enew = E_v_Evaluator.Energy(rho=new_rho, ions=ions, usePME=True)
    print('Energy New (a.u.)', Enew)
    print('Energy New (eV)', Enew * ENERGY_CONV['Hartree']['eV'])
    print('Energy New (eV/atom)', Enew * ENERGY_CONV['Hartree']['eV']/ions.nat)
    print('-' * 31, 'Time information', '-' * 31)
    print("{:28s}{:24s}{:20s}".format('Label', 'Cost(s)', 'Number'))
    for key in TimeData.cost :
        print("{:28s}{:<24.4f}{:<20d}".format(key, TimeData.cost[key], TimeData.number[key]))
    print('-' * 80)


if __name__ == "__main__":
    test_optim()
