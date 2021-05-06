import os
import time

import numpy as np


from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField
from dftpy.formats import npy
from dftpy.formats.xsf import XSF
from dftpy.functional import Functional
from dftpy.linear_solver import _get_atol
from dftpy.mpi import mp, sprint, MPIFile
from dftpy.td.casida import Casida
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.propagator import Propagator
from dftpy.utils import calc_rho, calc_j


def RealTimeRunner(config, rho0, E_v_Evaluator):
    outfile = config["TD"]["outfile"]
    int_t = config["TD"]["timestep"]
    t_max = config["TD"]["tmax"]
    max_pred_corr = config["TD"]["max_pc"]
    tol = config["TD"]["tol_pc"]
    atol = config["TD"]["atol_pc"]
    direc = config["TD"]["direc"]
    k = config["TD"]["strength"]
    max_runtime = config["TD"]["max_runtime"]
    restart = config["TD"]["restart"]
    correction = config["TD"]["correction"]
    correct_potential_name = config["TD"]["correct_potential"]
    vector_potential = config["TD"]['vector_potential']
    num_t = int(t_max / int_t)

    hamiltonian = Hamiltonian()
    prop = Propagator(hamiltonian, **config["PROPAGATOR"])

    if restart:
        fname = './tmp/restart_data.npy'
        if mp.size > 1:
            f = MPIFile(fname, mp, amode=mp.MPI.MODE_RDONLY)
        else:
            f = open(fname, "rb")
        i_t0 = npy.read(f, single=True) + 1
        psi = npy.read(f, grid=rho0.grid)
        psi = DirectField(grid=rho0.grid, rank=1, griddata_3d=psi, cplx=True)
        if vector_potential:
            A_t = npy.read(f, single=True)
            A_tm1 = npy.read(f, single=True)
    else:
        i_t0 = 0
        if vector_potential:
            psi = np.complex128(np.sqrt(rho0))
            A_t = np.zeros(3)
            A_t[direc] = k * SPEED_OF_LIGHT
            A_tm1 = A_t
        else:
            x = rho0.grid.r[direc]
            psi = np.sqrt(rho0) * np.exp(1j * k * x)
        psi.cplx = True

    N0 = rho0.integral()
    rho = calc_rho(psi)
    N = rho.integral()
    j = calc_j(psi)
    delta_rho = rho - rho0
    delta_mu = (delta_rho * delta_rho.grid.r).integral()
    j_int = j.integral()
    if vector_potential:
        Omega = psi.grid.Volume

    atol_rho = _get_atol(tol, atol, rho.norm())

    if correction:
        correct_potential = Functional(type='DYNAMIC', name=correct_potential_name)

    if not restart and mp.is_root:
        with open(outfile + "_mu", "w") as fmu:
            sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(delta_mu[0], delta_mu[1], delta_mu[2]), fileobj=fmu)
        with open(outfile + "_j", "w") as fj:
            sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(j_int[0], j_int[1], j_int[2]), fileobj=fj)
        with open(outfile + "_E", "w") as fE:
            pass
        if vector_potential:
            with open(outfile + "_A", "w") as fA:
                sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(A_t[0], A_t[1], A_t[2]), fileobj=fA)
        # if correction:
        #     with open(outfile + "_cor_mu", "w") as fmu:
        #         sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(delta_mu[0], delta_mu[1], delta_mu[2]),
        #                fileobj=fmu)
        #     with open(outfile + "_cor_j", "w") as fj:
        #         sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(j_int[0], j_int[1], j_int[2]), fileobj=fj)

    sprint("{:20s}{:30s}{:24s}".format('Iter', 'Num. of Predictor-corrector', 'Total Cost(s)'))
    begin_t = time.time()
    for i_t in range(i_t0, num_t):

        func = E_v_Evaluator.compute(rho, calcType=["V"], current=j)
        prop.hamiltonian.v = func.potential
        if vector_potential:
            prop.hamiltonian.A = A_t
        E = np.real(np.conj(psi) * prop.hamiltonian(psi)).integral()
        if vector_potential:
            E += Omega / 8.0 / np.pi / SPEED_OF_LIGHT ** 2 * (np.dot((A_t - A_tm1), (A_t - A_tm1)) / int_t / int_t)

        for i_pred_corr in range(max_pred_corr):
            if i_pred_corr > 0:
                old_rho_pred = rho_pred
                old_j_pred = j_pred
            else:
                atol_j = _get_atol(tol, atol, np.max(j.norm()))
            psi_pred, info = prop(psi, int_t)
            rho_pred = calc_rho(psi_pred)
            j_pred = calc_j(psi_pred)
            if vector_potential:
                A_t_pred = np.real(
                    2 * A_t - A_tm1 - 4 * np.pi * N * A_t / Omega * int_t * int_t - 4.0 * np.pi * SPEED_OF_LIGHT * N / Omega * psi_pred.para_current(
                        sigma=0.025) * int_t * int_t)

            if i_pred_corr > 0:
                diff_rho = (old_rho_pred - rho_pred).norm()
                diff_j = np.max((old_j_pred - j_pred).norm())
                if diff_rho < atol_rho and diff_j < atol_j:
                    break

            rho_corr = (rho + rho_pred) * 0.5
            j_corr = (j + j_pred) * 0.5
            func = E_v_Evaluator.compute(rho_corr, calcType=["V"], current=j_corr)
            prop.hamiltonian.v = func.potential
            if vector_potential:
                A_t_corr = (A_t + A_t_pred) * 0.5
                prop.hamiltonian.A = A_t_corr
        else:
            if max_pred_corr > 1:
                sprint('Convergence not reached for Predictor-corrector')
                if diff_rho >= atol_rho:
                    sprint('Diff in rho: {0:10.2e} > {1:10.2e}'.format(diff_rho, atol_rho))
                if diff_j >= atol_j:
                    sprint('Diff in j: {0:10.2e} > {1:10.2e}'.format(diff_j, atol_j))

        if correction:
            pot = correct_potential(rho, calcType=['V'], current=j).potential
            psi_pred = psi_pred + 1.0j * int_t * pot * psi
            psi_pred.normalize(N=N0)
            rho_pred = calc_rho(psi_pred)
            j_pred = calc_j(psi_pred)
            # delta_rho2 = rho_pred - rho0
            # delta_mu2 = (delta_rho2 * delta_rho2.grid.r).integral()
            # j2_int = j_pred.integral()
            #
            # if mp.is_root:
            #     with open(outfile + "_cor_mu", "a") as fmu:
            #         sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(delta_mu2[0], delta_mu2[1], delta_mu2[2]),
            #                fileobj=fmu)
            #     with open(outfile + "_cor_j", "a") as fj:
            #         sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(j2_int[0], j2_int[1], j2_int[2]), fileobj=fj)

        psi = psi_pred
        rho = rho_pred
        j = j_pred
        # print(psi_pred.para_current(sigma=0.025)[0], j_pred.integral()[0])
        if vector_potential:
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
            with open(outfile + "_E", "a") as fE:
                sprint("{0:17.10e}".format(E), fileobj=fE)
            if vector_potential:
                with open(outfile + "_A", "a") as fA:
                    sprint("{0:17.10e} {1:17.10e} {2:17.10e}".format(A_t[0], A_t[1], A_t[2]), fileobj=fA)

        cost_t = time.time() - begin_t
        sprint("{:<20d}{:<30d}{:<24.4f}".format(i_t + 1, i_pred_corr, cost_t))

        if info:
            break

        if max_runtime > 0 and cost_t > max_runtime:
            sprint('Maximum run time reached. Clean exiting.')
            if not os.path.isdir('./tmp'):
                os.mkdir('./tmp')
            fname = './tmp/restart_data.npy'
            if mp.size > 1:
                f = MPIFile(fname, mp, amode=mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
            else:
                f = open(fname, "wb")
            npy.write(f, i_t, single=True)
            npy.write(f, psi, grid=psi.grid)
            if vector_potential:
                npy.write(f, A_t, single=True)
                npy.write(f, A_tm1, single=True)
            break


def CasidaRunner(config, rho0, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    outfile = config["TD"]["outfile"]
    diagonalize = config["CASIDA"]["diagonalize"]
    tda = config["CASIDA"]["tda"]

    if diagonalize:
        potential = E_v_Evaluator(rho0, calcType={'V'}).potential
        hamiltonian = Hamiltonian(potential)
        sprint('Start diagonalizing Hamiltonian.')
        eigs, psi_list = hamiltonian.diagonalize(numeig)
        sprint('Diagonalizing Hamiltonian done.')
    else:
        raise Exception("diagonalize must be true.")

    E_v_Evaluator.UpdateFunctional(keysToRemove=['HARTREE', 'PSEUDO'])
    casida = Casida(rho0, E_v_Evaluator)

    sprint('Start building matrix.')
    casida.build_matrix(numeig, eigs, psi_list, build_ab=tda)
    sprint('Building matrix done.')

    if tda:
        omega, f = casida.tda()
    else:
        omega, f, x_minus_y_list = casida()

    with open(outfile, 'w') as fw:
        for i in range(len(omega)):
            fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))

    # if save_eigenvectors:
    #    if not os.isdir(ev_path):
    #        os.mkdir(ev_path)
    #    i = 0
    #    for x_minus_y in x_minus_y_list:
    #        with open('{0:s}/x_minus_y{1:d}',format(ev_path, i), 'w') as fw:


def DiagonalizeRunner(config, struct, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    eigfile = config["TD"]["outfile"]
    direct_to_psi = './xsf'

    potential = E_v_Evaluator(struct.field, calcType={'V'}).potential
    hamiltonian = Hamiltonian(potential)
    sprint('Start diagonalizing Hamiltonian.')
    eigs, psi_list = hamiltonian.diagonalize(numeig)
    sprint('Diagonalizing Hamiltonian done.')

    np.savetxt(eigfile, eigs, fmt='%15.8e')

    if not os.path.isdir(direct_to_psi):
        os.mkdir(direct_to_psi)
    for i in range(len(eigs)):
        XSF(filexsf='{0:s}/psi{1:d}.xsf'.format(direct_to_psi, i)).write(system=struct, field=psi_list[i])

# def SternheimerRunner(config, rho0, E_v_Evaluator):
#     outfile = config["TD"]["outfile"]
#
#     sternheimer = Sternheimer(rho0, E_v_Evaluator)
#     eigs, psi_list = sternheimer.hamiltonian.diagonalize(2)
#     sternheimer.grid.full = True
#     omega = np.linspace(0.0, 0.5, 26)
#     f = sternheimer(psi_list[1], omega, 0)
#     # f = omega
#     # sternheimer(psi_list[1], 1e-4, 0.01)
#
#     with open(outfile, 'w') as fw:
#         for i in range(len(omega)):
#             fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))

