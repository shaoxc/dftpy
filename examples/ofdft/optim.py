import numpy as np
from dftpy.formats.qepp import PP
from dftpy.optimization import Optimization
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.formats.qepp import PP
from dftpy.ewald import ewald
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import TimeData, bestFFTsize

def test_optim(self):
    path_pp='../DATA/'
    file1='Al_lda.oe01.recpot'
    posfile='fcc.vasp'
    ions = read_POSCAR(path_pos+posfile, names=['Al'])
    lattice = ions.pos.cell.lattice
    metric = np.dot(lattice.T, lattice)
    gap = 0.4
    nr = np.zeros(3, dtype = 'int32')
    for i in range(3):
        nr[i] = int(np.sqrt(metric[i, i])/gap)
    print('The initial grid size is ', nr)
    for i in range(3):
        nr[i] = bestFFTsize(nr[i])
    print('The final grid size is ', nr)
    grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=False)
    zerosA = np.zeros(grid.nnr, dtype=float)
    rho_ini = DirectField(grid=grid, griddata_F=zerosA, rank=1)
    optional_kwargs = {}
    optional_kwargs["PP_list"] = {'Al': path_pp+file1}
    optional_kwargs["ions"]    = ions 
    IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
    Vloc = IONS(rho_ini)
    ions.set_Zval()
    optional_kwargs = {}
    KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',optional_kwargs=optional_kwargs)
    XC = FunctionalClass(type='XC',name='LDA')
    HARTREE = FunctionalClass(type='HARTREE')

    charge_total = 0.0
    for i in range(ions.nat) :
        charge_total += ions.Zval[ions.labels[i]]
    rho_ini[:] = charge_total/ions.pos.cell.volume
    E_v_Evaluator = TotalEnergyAndPotential(rho=rho_ini,
                                    KineticEnergyFunctional=KE,
                                    XCFunctional=XC,
                                    HARTREE=HARTREE,
                                    IONS=IONS)
    optimization_options = {\
            'econv' : 1e-6, # Energy Convergence (a.u./atom)
            'maxfun' : 50,  # For TN method, it's the max steps for searching direction
            'maxiter' : 100,# The max steps for optimization
            }
    optimization_options["econv"] *= ions.nat
    opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_options = optimization_options, 
            optimization_method = 'TN')
    new_rho = opt.optimize_rho(guess_rho=rho_ini)
    print('Calc Energy')
    Enew = E_v_Evaluator.Energy(rho=new_rho, ions=ions, usePME = True)
    print('Energy New (a.u.)', Enew)
    print('Energy New (eV)', Enew * ENERGY_CONV['Hartree']['eV'])
    print('Energy New (eV/atom)', Enew * ENERGY_CONV['Hartree']['eV']/ions.nat)
    print('-' * 31, 'Time information', '-' * 31)
    print("{:28s}{:24s}{:20s}".format('Label', 'Cost(s)', 'Number'))
    for key in TimeData.cost :
        print("{:28s}{:<24.4f}{:<20d}".format(key, TimeData.cost[key], TimeData.number[key]))
    print('-' * 80)
