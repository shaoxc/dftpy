import numpy as np
from dftpy.functionals import TotalEnergyAndPotential
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.casida import Casida
# from dftpy.td.sternheimer import Sternheimer
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.system import System
from dftpy.utils import calc_rho, calc_j
from dftpy.dynamic_functionals_utils import DynamicPotential
from dftpy.formats.xsf import XSF
from dftpy.time_data import TimeData
import time
import os


def RealTimeRunner(config, rho0, E_v_Evaluator):

    outfile = config["TD"]["outfile"]
    int_t = config["PROPAGATOR"]["int_t"]
    t_max = config["TD"]["tmax"]
    order = config["TD"]["order"]
    direc = config["TD"]["direc"]
    k = config["TD"]["strength"]
    dynamic = config["TD"]["dynamic_potential"]
    max_runtime = config["TD"]["max_runtime"]
    restart = config["TD"]["restart"]
    num_t = int(t_max / int_t)

    hamiltonian = Hamiltonian()
    prop = Propagator(hamiltonian, interval=int_t, type=config["PROPAGATOR"]["type"], optional_kwargs=config["PROPAGATOR"])


    if restart:
        with open('./tmp/restart_data.npy', 'rb') as f:
            i_t0 = np.load(f)+1
            psi = np.load(f)
        psi = DirectField(grid=rho0.grid, rank=1, griddata_3d=psi, cplx=True)
    else:
        x = rho0.grid.r[direc]
        psi = np.sqrt(rho0) * np.exp(1j * k * x)
        psi.cplx = True
        i_t0 = 0

    rho = calc_rho(psi)
    j = calc_j(psi)
    delta_mu = np.empty(3)
    j_int = np.empty(3)
    delta_rho = rho - rho0
    delta_mu = (delta_rho * delta_rho.grid.r).integral()
    j_int = j.integral()
    eps = 1e-8

    if not restart:
        with open(outfile + "_mu", "w") as fmu:
            fmu.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(delta_mu[0], delta_mu[1], delta_mu[2]))
        with open(outfile + "_j", "w") as fj:
            fj.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(j_int[0], j_int[1], j_int[2]))
        with open(outfile + "_E", "w") as fE:
            pass

    print("{:20s}{:30s}{:24s}".format('Iter', 'Num. of Predictor-corrector', 'Total Cost(s)'))
    begin_t = time.time()
    for i_t in range(i_t0, num_t):
        t = int_t * i_t
        func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType=["V"])
        prop.hamiltonian.v = func.potential
        if dynamic:
            prop.hamiltonian.v += DynamicPotential(rho, j)
        E = np.real(np.conj(psi) * prop.hamiltonian(psi)).integral()

        for i_cn in range(order):
            if i_cn > 0:
                old_rho1 = rho1
                old_j1 = j1
            psi1, info = prop(psi)
            rho1 = calc_rho(psi1)
            j1 = calc_j(psi1)
            if i_cn > 0 and np.max(np.abs(old_rho1 - rho1)) < eps and np.max(np.abs(old_j1 - j1)) < eps:
                break

            rho_half = (rho + rho1) * 0.5
            func = E_v_Evaluator.ComputeEnergyPotential(rho_half, calcType=["V"])
            prop.hamiltonian.v = func.potential
            if dynamic:
                j_half = (j + j1) * 0.5
                prop.hamiltonian.v += DynamicPotential(rho_half, j_half)

        psi = psi1
        rho = rho1
        j = j1

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        j_int = j.integral()

        with open(outfile + "_mu", "a") as fmu:
            fmu.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(delta_mu[0], delta_mu[1], delta_mu[2]))
        with open(outfile + "_j", "a") as fj:
            fj.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(j_int[0], j_int[1], j_int[2]))
        with open(outfile + "_E", "a") as fE:
            fE.write("{0:17.10e}\n".format(E))

        cost_t = time.time() - begin_t
        print("{:<20d}{:<30d}{:<24.4f}".format(i_t+1, i_cn, cost_t))
        if info:
            break
        if max_runtime > 0 and cost_t > max_runtime:
            print('Maximum run time reached. Clean exitting.')
            if not os.path.isdir('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/restart_data.npy', 'wb') as f:
                np.save(f, i_t)
                np.save(f, psi)
            break


def CasidaRunner(config, rho0, E_v_Evaluator):

    numeig = config["CASIDA"]["numeig"]
    outfile = config["TD"]["outfile"]
    diagonize = config["CASIDA"]["diagonize"]
    tda = config["CASIDA"]["tda"]

    if diagonize:
        potential = E_v_Evaluator(rho0, calcType=['V']).potential
        hamiltonian = Hamiltonian(potential)
        print('Start diagonizing Hamlitonian.')
        eigs, psi_list = hamiltonian.diagonize(numeig)
        print('Diagonizing Hamlitonian done.')
    else:
        raise Exception("diagonize must be true.")

    E_v_Evaluator.UpdateFunctional(keysToRemove = ['HARTREE', 'PSEUDO'])
    casida = Casida(rho0, E_v_Evaluator)

    print('Start buildling matrix.')
    casida.build_matrix(numeig, eigs, psi_list, build_ab = tda)
    print('Building matrix done.')

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
    print('Start diagonizing Hamlitonian.')
    eigs, psi_list = hamiltonian.diagonize(numeig)
    print('Diagonizing Hamlitonian done.')

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
