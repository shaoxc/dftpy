import os
import os.path

import numpy as np

from dftpy.formats import io
from dftpy.mpi import sprint
from dftpy.td.casida import Casida
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.sternheimer import Sternheimer
from dftpy.field import DirectField


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


def DiagonalizeRunner(config, field, ions, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    eigfile = config["TD"]["outfile"]
    direct_to_psi = './xsf'

    potential = E_v_Evaluator(field, calcType={'V'}).potential
    hamiltonian = Hamiltonian(potential)
    sprint('Start diagonalizing Hamiltonian.')
    eigs, psi_list = hamiltonian.diagonalize(numeig)
    sprint('Diagonalizing Hamiltonian done.')

    np.savetxt(eigfile, eigs, fmt='%15.8e')

    if not os.path.isdir(direct_to_psi):
        os.mkdir(direct_to_psi)
    for i in range(len(eigs)):
        io.write('{0:s}/psi{1:d}.xsf'.format(direct_to_psi, i), ions, psi_list[i])


def osillator_strength(drho: DirectField, e: float) -> float:
    grid = drho.grid
    result = 0.0
    for direc in range(3):
        x = grid.r[direc]
        result += (x * drho).integral()

    result *= -2.0 / e
    return result


def SternheimerRunner(config, rho0, E_v_Evaluator):
    outfile = config["TD"]["outfile"]

    hamiltonian = Hamiltonian(v=E_v_Evaluator(rho0, calcType=['V']).potential)
    eigs, psi_list = hamiltonian.diagonalize(2)
    drho0 = psi_list[0]*psi_list[1]*rho0.integral()
    # r = DirectField(grid=system.field.grid, griddata_3d=(system.field.grid.r[0] - system.field.grid.lattice[0][0] / 2))
    # print(r.integral())
    # diff = np.zeros_like(system.field)
    # diff[1::, 1::, 1::] = system.field[1::, 1::, 1::] - system.field[-1:0:-1, -1:0:-1, -1:0:-1]
    # print(np.abs(diff).integral())
    # from dftpy.formats.xsf import XSF
    # xsf = XSF(filexsf='r.xsf')
    # xsf.write(system=system, field=r)

    e = 1e-3
    #omega_list = np.linspace(0.05, 0.5, 10)
    omega_list = [0.1]
    oscillator_strength_list = []
    with open(outfile, 'w') as fw:
        pass
    for omega in omega_list:
        sprint('omega: ', omega)
        sternheimer = Sternheimer(rho0, functionals=E_v_Evaluator, drho0=drho0, omega=omega, e0=eigs[0], e=e, outfile=outfile)
        sternheimer()
        dv = sternheimer.calc_dv(sternheimer.drho)
        sternheimer.sternheimer_equation_solver.dv = dv
        drho = sternheimer.sternheimer_equation_solver() * sternheimer.N
        sprint('drho_diff: ', np.abs(drho-sternheimer.drho).integral())
        #sprint(drho)
        sprint('drho: ', sternheimer.drho.integral())
        osc = osillator_strength(sternheimer.drho, e)
        oscillator_strength_list.append(osc)
        with open(outfile, 'a') as fw:
            fw.write('{0:17.10e} {1:17.10e}\n'.format(omega, osc))

    # f = omega
    # sternheimer(psi_list[1], 1e-4, 0.01)

    # with open(outfile, 'w') as fw:
    #     for i in range(len(omega)):
    #         fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], oscillator_strength_list[i]))
