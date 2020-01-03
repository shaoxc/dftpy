import numpy as np
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.propagator import Propagator, hamiltonian
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.system import System
import time


def cal_rho_j(psi):
    rho = np.real(psi * np.conj(psi))
    s = DirectField(psi.grid, rank=1, griddata_3d=np.angle(psi))
    j = np.real(rho * s.gradient())
    return rho, j


def tdrunner(rho0, E_v_Evaluator, config):

    outfile = config["TD"]["outfile"]
    int_t = config["TD"]["int_t"]
    t_max = config["TD"]["tmax"]
    order = config["TD"]["order"]
    direc = config["TD"]["direc"]
    num_t = int(t_max / int_t)

    prop = Propagator(interval=int_t, type="crank-nicolson", optional_kwargs=config["PROPAGATOR"])

    begin_t = time.time()
    x = rho0.grid.r[direc]
    x = np.expand_dims(x, 3)
    k = 1.0e-6
    psi = np.sqrt(rho0) * np.exp(1j * k * x)
    rho, j = cal_rho_j(psi)
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
        func = E_v_Evaluator.ComputeEnergyPotential(rho, calcType="Potential")
        potential = func.potential
        E = np.real(np.conj(psi) * hamiltonian(psi, potential)).integral()

        for i_cn in range(order):
            if i_cn > 0:
                old_rho1 = rho1
                old_j1 = j1
            psi1, info = prop(psi, potential)
            rho1, j1 = cal_rho_j(psi1)
            if i_cn > 0 and np.max(np.abs(old_rho1 - rho1)) < eps and np.max(np.abs(old_j1 - j1)) < eps:
                print(i_cn)
                break

            rho_half = (rho + rho1) * 0.5
            func = E_v_Evaluator.ComputeEnergyPotential(rho_half, calcType="Potential")
            potential = func.potential

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