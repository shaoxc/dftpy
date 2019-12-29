import numpy as np
from dftpy.optimization import Optimization
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.formats.vasp import read_POSCAR
import argparse

def read_input(inptfile):
    with open(inptfile, 'r') as f:
        for line in f:
            tmp = line.rstrip('\n').split()
            if tmp[0] == 'posfile':
                posfile = tmp[1]
            elif tmp[0] == 'pseudodir':
                pseudo_dir = tmp[1]
            elif tmp[0] == 'intv':
                intv = float(tmp[1])
            elif tmp[0] == 'natom':
                natom = int(tmp[1])
            elif tmp[0] == 'atomic_species':
                atom_name = []
                atom_zval = []
                atom_pseudo = []
                for i_atom in range(natom):
                    tmp = f.readline().rstrip('\n').split()
                    atom_name.append(tmp[0])
                    atom_zval.append(float(tmp[1]))
                    atom_pseudo.append(pseudo_dir + tmp[2])

    return posfile, intv, atom_name, atom_zval, atom_pseudo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='test.in', dest='infile')
    args = parser.parse_args()
    
    posfile, intv, atom_name, atom_zval, atom_pseudo = read_input(args.infile)
    
    ions = read_POSCAR('/home/kj385/input/'+posfile, names=atom_name)
    ions.Zval = dict(zip(atom_name, atom_zval))
    lattice = ions.pos.cell.lattice
    nr = np.empty(3, dtype = 'int32')
    for i in range(3):
        nr[i] = np.ceil(lattice[i,i]/intv)
    print('The grid size is ', nr)
    grid = DirectGrid(lattice=lattice, nr=nr, units=None)
    n_tot = 0.0
    for i in range(ions.nat):
        n_tot += ions.Zval[ions.labels[i]]

    #mol.field = mol.field/n_tot_mol*n_tot
    rho_ini = DirectField(grid=grid, rank=1)
    rho_ini[:] = n_tot/grid.Volume
    
    optional_kwargs = {}
    optional_kwargs["Sigma"] = 0.025
    optional_kwargs["x"] = 1.0
    optional_kwargs["y"] = 1.0
    KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=False,optional_kwargs=optional_kwargs)
    #KE = FunctionalClass(type='KEDF',name='TF',is_nonlocal=False,optional_kwargs=optional_kwargs)

    XC = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)

    optional_kwargs = {}
    optional_kwargs["PP_list"] = dict(zip(atom_name, atom_pseudo))
    optional_kwargs["ions"] = ions

    PSEUDO = FunctionalClass(type='PSEUDO', optional_kwargs=optional_kwargs)
    HARTREE = FunctionalClass(type='HARTREE')

    E_v_Evaluator = TotalEnergyAndPotential(rho=rho_ini,
                                            KineticEnergyFunctional=KE,
                                            XCFunctional=XC,
                                            HARTREE=HARTREE,
                                            PSEUDO=PSEUDO)
    
    optional_kwargs = {}
    optional_kwargs['econv'] = 1.0e-11
    optional_kwargs['ftol'] = 1.0e-14
    optional_kwargs['gtol'] = 1.0e-14
    optional_kwargs['maxiter'] = 100
    #opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'LBFGS')
    opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'TN', optimization_options=optional_kwargs)
    #opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'CG-HS')

    new_rho = opt.optimize_rho(guess_rho=rho_ini)
    
    np.save('./density', new_rho)
    with open('./grid', 'w') as f:
        f.write('{0:d} {1:d} {2:d}\n'.format(nr[0], nr[1], nr[2]))
        for i in range(3):
            f.write('{0:15.8e} {1:15.8e} {2:15.8e}\n'.format(lattice[i][0], lattice[i][1], lattice[i][2]))

main()
