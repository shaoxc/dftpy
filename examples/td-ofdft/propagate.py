import numpy as np
from dftpy.formats.qepp import PP
from dftpy.formats.xsf import XSF
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.dynamic_functionals_utils import DynamicPotential
from dftpy.propagator import Propagator, hamiltonian
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.formats.vasp import read_POSCAR
from dftpy.system import System
import time
import argparse

def v_ext(x, t, k, omega):
    return k*np.cos(omega*t)*x

def cal_rho_j(psi):
    rho = np.real(psi*np.conj(psi))
    s = DirectField(psi.grid, rank=1, griddata_3d=np.angle(psi))
    j = np.real(rho*s.gradient())
    return rho,j

def read_input(inptfile):
    with open(inptfile, 'r') as f:
        for line in f:
            tmp = line.rstrip('\n').split()
            if tmp[0] == 'outfile':
                outfile = tmp[1]
            elif tmp[0] == 'posfile':
                posfile = tmp[1]
            elif tmp[0] == 'int_t':
                int_t = float(tmp[1])
            elif tmp[0] == 'tmax':
                t_max = float(tmp[1])
            elif tmp[0] == 'order':
                order = int(tmp[1])
            elif tmp[0] == 'pseudodir':
                pseudo_dir = tmp[1]
            elif tmp[0] == 'direc':
                direc = int(tmp[1])
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



    return outfile, posfile, int_t, t_max, order, direc, atom_name, atom_zval, atom_pseudo

def main():
    np.set_printoptions(threshold = np.inf)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='test.in', dest='infile')
    args = parser.parse_args()
    
    outfile, posfile, int_t, t_max, order, direc, atom_name, atom_zval, atom_pseudo = read_input(args.infile)

    num_t = int(t_max/int_t)

    with open('./grid','r') as f:
        nr = list(map(int, f.readline().split()))
        lattice = []
        for i in range(3):
            lattice.append(list(map(float, f.readline().split())))

    lattice=np.asarray(lattice)

    grid = DirectGrid(lattice=lattice, nr=nr, units=None)
    rho0 = DirectField(grid=grid, rank=1, griddata_3d=np.load('./density.npy'))
    if np.isnan(rho0).any():
        print('rho0 bad')

    ions = read_POSCAR('/home/kj385/input/'+posfile, names=atom_name)
    ions.Zval = dict(zip(atom_name, atom_zval))
    mol = System(ions, grid, field = rho0)
    XSF(filexsf='./xsf/rho.xsf').write(system=mol)

    # return 0;

    #KE
    optional_kwargs = {}
    #KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=False,optional_kwargs=optional_kwargs)
    KE = FunctionalClass(type='KEDF',name='TF',is_nonlocal=False,optional_kwargs=optional_kwargs)
    #KE = FunctionalClass(type='KEDF',name='vW',is_nonlocal=False,optional_kwargs=optional_kwargs)

    #XC
    XC = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)

    #Hartree
    optional_kwargs = {}
    optional_kwargs["PP_list"] = dict(zip(atom_name, atom_pseudo))
    optional_kwargs["ions"]    = ions

    PSEUDO = FunctionalClass(type='PSEUDO', optional_kwargs=optional_kwargs)
    HARTREE = FunctionalClass(type='HARTREE')

    E_v_Evaluator = TotalEnergyAndPotential(KineticEnergyFunctional=KE,
                                            XCFunctional=XC,
                                            HARTREE=HARTREE,
                                            PSEUDO=PSEUDO,
                                            rho=rho0)

    optional_kwargs = {}
    optional_kwargs['order'] = 20
    optional_kwargs['linearsolver'] = 'bicgstab'
    #optional_kwargs['linearsolver'] = 'lgmres'
    optional_kwargs['tol'] = 1e-10
    optional_kwargs['maxiter'] = 100
    optional_kwargs['sigma'] = 0.025
    #prop = Propagator(interval = int_t, type='taylor', optional_kwargs=optional_kwargs)
    prop = Propagator(interval = int_t, type='crank-nicolson', optional_kwargs=optional_kwargs)

    begin_t = time.time()
    x = rho0.grid.r[:,:,:,direc]
    x=np.expand_dims(x,3)
    k = 1.0e-6
    psi = np.sqrt(rho0)*np.exp(1j*k*x)
    #psi = np.sqrt(rho0)
    rho, j = cal_rho_j(psi)
    delta_mu = np.empty(3)
    j_int = np.empty(3)
    delta_rho = rho-rho0
    delta_mu=(delta_rho*delta_rho.grid.r).integral()
    j_int=j.integral()

    eps = 1e-8
    with open('./'+outfile+'_mu','w') as fmu:
        fmu.write('{0:17.10e} {1:17.10e} {2:17.10e}\n'.format(delta_mu[0], delta_mu[1], delta_mu[2]))
    with open('./'+outfile+'_j','w') as fj:
        fj.write('{0:17.10e} {1:17.10e} {2:17.10e}\n'.format(j_int[0], j_int[1], j_int[2]))
    with open('./'+outfile+'_E','w') as fE:
        pass

    for i_t in range(num_t):
        cost_t = time.time()-begin_t
        print('iter: {0:d} time: {1:f}'.format(i_t, cost_t))
        t = int_t * i_t
        func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType='Potential')
        #potential0 = func.potential
        #potential0 = potential0 + DynamicPotential(rho, j)
        #potential0 = potential0 + v_ext(x, t, k, omega)
        potential = func.potential
        E = np.real(np.conj(psi)*hamiltonian(psi,potential)).integral()

        #potential = potential0
        for i_cn in range(order):
            if i_cn > 0:
                old_rho1 = rho1
                old_j1 = j1
            psi1, info = prop(psi, potential)
            rho1, j1 = cal_rho_j(psi1)
            if i_cn > 0 and np.max(np.abs(old_rho1-rho1))<eps and np.max(np.abs(old_j1-j1))<eps:
                print(i_cn)
                break

            rho_half = (rho + rho1) * 0.5
            func = E_v_Evaluator.ComputeEnergyPotential(rho_half, calcType='Potential')
            #func = E_v_Evaluator.ComputeEnergyPotential(rho1, calcType='Potential')
            #potential1 = func.potential
            #potential1 = potential1 + DynamicPotential(rho1, j1)
            #potential1 = potential1 + v_ext(x, t+int_t, k, omega)
            #potential = (potential0+potential1) * 0.5
            potential = func.potential

        psi = psi1
        rho = rho1
        j = j1
        #if False and i_t%100 == 0:
        #    XSF(filexsf='./xsf/rho{0:d}.xsf'.format(i_t)).write(system=mol, field=rho)
        #    XSF(filexsf='./xsf/v{0:d}.xsf'.format(i_t)).write(system=mol, field=potential)

        delta_rho = rho-rho0
        delta_mu=(delta_rho*delta_rho.grid.r).integral()
        j_int=j.integral()
        
        with open('./'+outfile+'_mu','a') as fmu:
            fmu.write('{0:17.10e} {1:17.10e} {2:17.10e}\n'.format(delta_mu[0], delta_mu[1], delta_mu[2]))
        with open('./'+outfile+'_j','a') as fj:
            fj.write('{0:17.10e} {1:17.10e} {2:17.10e}\n'.format(j_int[0], j_int[1], j_int[2]))
        with open('./'+outfile+'_E','a') as fE:
            fE.write('{0:17.10e}\n'.format(E))
        
        if info:
            break

    #np.savetxt('./'+outfile+'_mu', np.transpose(delta_mu), fmt='%17.10e')
    #np.savetxt('./'+outfile+'_j', np.transpose(j_int), fmt='%17.10e')
    #np.savetxt('./'+outfile+'_E', E, fmt='%17.10e')


main()
