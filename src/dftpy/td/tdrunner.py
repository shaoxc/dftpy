import numpy as np
from dftpy.functionals import TotalEnergyAndPotential
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamitonian
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.system import System
from dftpy.utils import calc_rho, calc_j
import time


def tdrunner(rho0, E_v_Evaluator, config):

    outfile = config["TD"]["outfile"]
    int_t = config["PROPAGATOR"]["int_t"]
    t_max = config["TD"]["tmax"]
    order = config["TD"]["order"]
    direc = config["TD"]["direc"]
    k = config["TD"]["strength"]
    num_t = int(t_max / int_t)

    hamiltonian = Hamiltonian()
    prop = Propagator(hamiltonian, interval=int_t, type=config["PROPAGATOR"]["type"], optional_kwargs=config["PROPAGATOR"])

    begin_t = time.time()
    x = rho0.grid.r[direc]
    psi = np.sqrt(rho0) * np.exp(1j * k * x)
    psi.cplx = True
    rho = calc_rho(psi)
    j = calc_j(psi)
    delta_mu = np.empty(3)
    j_int = np.empty(3)
    delta_rho = rho - rho0
    delta_mu = (delta_rho * delta_rho.grid.r).integral()
    j_int = j.integral()

    eps = 1e-8
    with open("./" + outfile + "_mu", "w") as fmu:
        fmu.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(delta_mu[0], delta_mu[1], delta_mu[2]))
    with open("./" + outfile + "_j", "w") as fj:
        fj.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(j_int[0], j_int[1], j_int[2]))
    with open("./" + outfile + "_E", "w") as fE:
        pass

    for i_t in range(num_t):
        cost_t = time.time() - begin_t
        print("iter: {0:d} time: {1:f}".format(i_t, cost_t))
        t = int_t * i_t
        func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType=["V"])
        prop.hamitonian.v = func.potential
        E = np.real(np.conj(psi) * prop.hamiltonian(psi)).integral()

        for i_cn in range(order):
            if i_cn > 0:
                old_rho1 = rho1
                old_j1 = j1
            psi1, info = prop(psi)
            rho1 = calc_rho(psi1)
            j1 = calc_j(psi1)
            if i_cn > 0 and np.max(np.abs(old_rho1 - rho1)) < eps and np.max(np.abs(old_j1 - j1)) < eps:
                print(i_cn)
                break

            rho_half = (rho + rho1) * 0.5
            func = E_v_Evaluator.ComputeEnergyPotential(rho_half, calcType=["V"])
            prop.hamiltonian.v = func.potential

        psi = psi1
        rho = rho1
        j = j1

        delta_rho = rho - rho0
        delta_mu = (delta_rho * delta_rho.grid.r).integral()
        j_int = j.integral()

        with open("./" + outfile + "_mu", "a") as fmu:
            fmu.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(delta_mu[0], delta_mu[1], delta_mu[2]))
        with open("./" + outfile + "_j", "a") as fj:
            fj.write("{0:17.10e} {1:17.10e} {2:17.10e}\n".format(j_int[0], j_int[1], j_int[2]))
        with open("./" + outfile + "_E", "a") as fE:
            fE.write("{0:17.10e}\n".format(E))

        if info:
            break
