import numpy as np
from dftpy.mpi import mp, sprint, MPIFile
from dftpy.field import DirectField
from dftpy.functionals import TotalEnergyAndPotential
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.casida import Casida
# from dftpy.td.sternheimer import Sternheimer
from dftpy.system import System
from dftpy.utils import calc_rho, calc_j
from dftpy.dynamic_functionals_utils import DynamicPotential
from dftpy.formats.xsf import XSF
from dftpy.formats import npy
from dftpy.time_data import TimeData
from dftpy.constants import SPEED_OF_LIGHT
from dftpy.linear_solver import _get_atol
import time
#import tracemalloc
import os


def RealTimeRunner(config, rho0, E_v_Evaluator):
    #tracemalloc.start()

    outfile = config["TD"]["outfile"]
    int_t = config["TD"]["timestep"]
    t_max = config["TD"]["tmax"]
    max_pred_corr = config["TD"]["max_pc"]
    tol = config["TD"]["tol_pc"]
    atol = config["TD"]["atol_pc"]
    direc = config["TD"]["direc"]
    k = config["TD"]["strength"]
    dynamic = config["TD"]["dynamic_potential"]
    max_runtime = config["TD"]["max_runtime"]
    restart = config["TD"]["restart"]
    num_t = int(t_max / int_t)

    hamiltonian = Hamiltonian()
    prop = Propagator(hamiltonian, **config["PROPAGATOR"])

    #E_v_Evaluator.UpdateFunctional(keysToRemove=["PSEUDO"])
    #print(E_v_Evaluator.funcDict)

    if restart:
        fname = './tmp/restart_data.npy'
        if mp.size > 1 :
            f = MPIFile(fname, mp, amode = mp.MPI.MODE_RDONLY)
        else :
            f = open(fname, "rb")
        i_t0 = npy.read(f, single = True)+1
        psi = npy.read(f, grid = rho0.grid)
        psi = DirectField(grid=rho0.grid, rank=1, griddata_3d=psi, cplx=True)
    else:
        #x = rho0.grid.r[direc]
        psi = np.sqrt(rho0)
        psi.cplx = True
        i_t0 = 0
        A_t = np.zeros(3)
        A_t[direc]=k*SPEED_OF_LIGHT
        #A_tm1 = A_t * (1-int_t)
        A_tm1 = A_t

    rho = calc_rho(psi)
    N = rho.integral()
    j = calc_j(psi)
    delta_mu = np.empty(3)
    j_int = np.empty(3)
    delta_rho = rho - rho0
    delta_mu = (delta_rho * delta_rho.grid.r).integral()
    j_int = j.integral()
    Omega = psi.grid.Volume
    atol_rho = _get_atol(tol, atol, rho.norm())

    if not restart:
        if mp.is_root:
            with open(outfile + "_mu", "w") as fmu:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(delta_mu[0], delta_mu[1], delta_mu[2]), fileobj=fmu)
            with open(outfile + "_j", "w") as fj:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(j_int[0], j_int[1], j_int[2]), fileobj=fj)
            with open(outfile + "_A", "w") as fA:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(A[0], A[1], A[2]), fileobj=fA)
            with open(outfile + "_E", "w") as fE:
                pass

    sprint("{:20s}{:30s}{:24s}".format('Iter', 'Num. of Predictor-corrector', 'Total Cost(s)'))
    begin_t = time.time()
    for i_t in range(i_t0, num_t):
        
        #current_mem, peak_mem = tracemalloc.get_traced_memory()
        #print(f"Current memory usage is {current_mem / 1.0e6}MB; Peak was {peak_mem / 1.0e6}MB")

        t = int_t * i_t
        func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType=["V"])
        prop.hamiltonian.v = func.potential
        if dynamic:
            prop.hamiltonian.v += DynamicPotential(rho, j)
        prop.hamiltonian.A = A_t
        E = np.real(np.conj(psi) * prop.hamiltonian(psi)).integral()
        E += Omega/8.0/np.pi/SPEED_OF_LIGHT**2*(np.dot((A_t-A_tm1),(A_t-A_tm1))/int_t/int_t)

        for i_pred_corr in range(max_pred_corr):
            if i_pred_corr > 0:
                old_rho_pred = rho_pred
                old_j_pred = j_pred
                old_A_t_pred = A_t_pred
            else:
                atol_j = _get_atol(tol, atol, np.max(j.norm()))
                atol_A = _get_atol(tol, atol, np.max(A_t.norm()))
            psi_pred, info = prop(psi, int_t)
            rho_pred = calc_rho(psi_pred)
            j_pred = calc_j(psi_pred)
            #A_t_pred = np.real(2*A_t - A_tm1 + 4*np.pi*N*A_t/Omega*int_t*int_t + 4.0j*np.pi*SPEED_OF_LIGHT*N/Omega*(np.conj(psi_pred)*psi_pred.gradient(flag = "supersmooth", force_real=False)).integral())
            A_t_pred = np.real(2*A_t - A_tm1 - 4*np.pi*N*A_t/Omega*int_t*int_t - 4.0*np.pi*SPEED_OF_LIGHT*N/Omega*psi_pred.para_current(sigma=0.025)*int_t*int_t)
            #A_t_pred = np.real(2*A_t - A_tm1 + 4.0*np.pi*SPEED_OF_LIGHT*N/Omega*psi_pred.para_current(sigma=0.025)*int_t*int_t)

            if i_pred_corr > 0 and (old_rho_pred - rho_pred).norm() < atol_rho and np.max((old_j_pred - j_pred).norm()) < atol_j and np.max((old_A_t_pred - A_t_pred).norm()) < atol_A:
                break

            rho_corr = (rho + rho_pred) * 0.5
            func = E_v_Evaluator.ComputeEnergyPotential(rho_corr, calcType=["V"])
            prop.hamiltonian.v = func.potential
            if dynamic:
                j_corr = (j + j_pred) * 0.5
                prop.hamiltonian.v += DynamicPotential(rho_corr, j_corr)
            A_t_corr = (A_t + A_t_pred) * 0.5
            prop.hamiltonian.A = A_t_corr

        psi = psi_pred
        rho = rho_pred
        j = j_pred
        #print(psi_pred.para_current(sigma=0.025)[0], j_pred.integral()[0])
        A_tm1 = A_t
        A_t = A_t_pred

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        j_int = j.integral()

        if mp.is_root:
            with open(outfile + "_mu", "a") as fmu:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(delta_mu[0], delta_mu[1], delta_mu[2]), fileobj=fmu)
            with open(outfile + "_j", "a") as fj:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(j_int[0], j_int[1], j_int[2]), fileobj=fj)
            with open(outfile + "_A", "a") as fA:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(A[0], A[1], A[2]), fileobj=fA)
            with open(outfile + "_E", "a") as fE:
                sprint("{0:17.10e}".format(E), fileobj=fE)

        cost_t = time.time() - begin_t
        sprint("{:<20d}{:<30d}{:<24.4f}".format(i_t+1, i_pred_corr, cost_t))
        
        if info:
            break

        if max_runtime > 0 and cost_t > max_runtime:
            sprint('Maximum run time reached. Clean exitting.')
            if not os.path.isdir('./tmp'):
                os.mkdir('./tmp')
            fname = './tmp/restart_data.npy'
            if mp.size > 1:
                f = MPIFile(fname, mp, amode = mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
            else :
                f = open(fname, "wb")
            npy.write(f, i_t, single = True)
            npy.write(f, psi, grid = psi.grid)
            break

    #tracemalloc.stop()


def CasidaRunner(config, rho0, E_v_Evaluator):

    numeig = config["CASIDA"]["numeig"]
    outfile = config["TD"]["outfile"]
    diagonize = config["CASIDA"]["diagonize"]
    tda = config["CASIDA"]["tda"]

    if diagonize:
        potential = E_v_Evaluator(rho0, calcType=['V']).potential
        hamiltonian = Hamiltonian(potential)
        sprint('Start diagonizing Hamlitonian.')
        eigs, psi_list = hamiltonian.diagonize(numeig)
        sprint('Diagonizing Hamlitonian done.')
    else:
        raise Exception("diagonize must be true.")

    E_v_Evaluator.UpdateFunctional(keysToRemove = ['HARTREE', 'PSEUDO'])
    casida = Casida(rho0, E_v_Evaluator)

    sprint('Start buildling matrix.')
    casida.build_matrix(numeig, eigs, psi_list, build_ab = tda)
    sprint('Building matrix done.')

    if tda:
        omega, f = casida.tda()
    else:
        omega, f, x_minus_y_list = casida()

    with open(outfile, 'w') as fw:
        for i in range(len(omega)):
            fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))

    #if save_eigenvectors:
    #    if not os.isdir(ev_path):
    #        os.mkdir(ev_path)
    #    i = 0
    #    for x_minus_y in x_minus_y_list:
    #        with open('{0:s}/x_minus_y{1:d}',format(ev_path, i), 'w') as fw:


def DiagonizeRunner(config, struct, E_v_Evaluator):

    numeig = config["CASIDA"]["numeig"]
    eigfile = config["TD"]["outfile"]
    direct_to_psi = './xsf'

    potential = E_v_Evaluator(struct.field, calcType=['V']).potential
    hamiltonian = Hamiltonian(potential)
    sprint('Start diagonizing Hamlitonian.')
    eigs, psi_list = hamiltonian.diagonize(numeig)
    sprint('Diagonizing Hamlitonian done.')

    np.savetxt(eigfile, eigs, fmt='%15.8e')

    if not os.path.isdir(direct_to_psi):
        os.mkdir(direct_to_psi)
    for i in range(len(eigs)):
        XSF(filexsf='{0:s}/psi{1:d}.xsf'.format(direct_to_psi, i)).write(system=struct, field=psi_list[i])





def SternheimerRunner(config, rho0, E_v_Evaluator):

    outfile = config["TD"]["outfile"]

    sternheimer = Sternheimer(rho0, E_v_Evaluator)
    eigs, psi_list = sternheimer.hamiltonian.diagonize(2)
    sternheimer.grid.full = True
    omega = np.linspace(0.0, 0.5, 26)
    f = sternheimer(psi_list[1], omega, 0)
    #f = omega
    #sternheimer(psi_list[1], 1e-4, 0.01)

    with open(outfile, 'w') as fw:
        for i in range(len(omega)):
            fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))
