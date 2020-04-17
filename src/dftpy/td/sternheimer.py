import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as linalg
from scipy.optimize import minimize
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.utils import calc_rho, calc_drho
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.td.hamiltonian import Hamiltonian

class Sternheimer(object):

    def __init__(self, rho0, E_v_Evaluator):
        self.rho0 = rho0
        self.N = rho0.integral()
        self.psi0 = np.sqrt(rho0/self.N)
        self.functional = E_v_Evaluator
        self.hamiltonian = Hamiltonian(v = self.functional(self.rho0, calcType=['V']).potential)
        self.kxc = E_v_Evaluator.Subset(['KineticEnergyFunctional','XCFunctional'])
        self.grid = rho0.grid
        self.e0 = (self.psi0 * self.hamiltonian(self.psi0)).integral()


    def calc_dv(self, dpsi):
        drho = calc_drho(self.psi0, dpsi, self.N)
        ftxc = self.functional(self.rho0, calcType = ['V2']).v2rho2
        dvtxc = ftxc * drho
        hartree = FunctionalClass(type='HARTREE')
        dvh = hartree(drho, calcType = ['V']).potential
        return dvtxc+dvh


    def matvecUtil(self, omega, eta):
        def matvec(dpsi_):
            dpsi = DirectField(grid=self.grid, rank=1, griddata_3d=np.reshape(dpsi_, self.grid.nr), cplx=True)
            prod = self.hamiltonian(dpsi) + ( -self.e0 + omega + 1j * eta) * dpsi
            return prod.ravel()

        return matvec

    def linearSolver(self, dpsi_old, dv, omega, eta, linearsolver="bicgstab", tol=1e-8, maxiter=1000):

        LinearSolverDict = {
            "bicg": linalg.bicg,
            "bicgstab": linalg.bicgstab,
            "cg": linalg.cg,
            "cgs": linalg.cgs,
            "gmres": linalg.gmres,
            "lgmres": linalg.lgmres,
            "minres": linalg.minres,
            "qmr": linalg.qmr,
        }

        b = (-dv * self.psi0 + (self.psi0 * dv * self.psi0).integral() * self.psi0).ravel()
        size = self.grid.nnr
        A = LinearOperator((size, size), dtype='complex128', matvec=self.matvecUtil(omega, eta))
        try:
            dpsi_, info = LinearSolverDict[linearsolver](A, b, x0=dpsi_old.ravel(), tol=tol, maxiter=maxiter, atol=0)
        except KeyError:
            raise AttributeError("{0:s} is not a linear solver".format(linearsolver))

        if info:
            print(info)
        dpsi = DirectField(grid=self.grid, rank=1, griddata_3d=np.reshape(dpsi_, self.grid.nr), cplx=True)
        
        return dpsi, info

    def scf(self, dpsi_ini, omega, eta):
        #print((psi1*psi1).integral())
        #dpsi = psi1 - self.psi*(self.psi*psi1).integral()
        #print((dpsi*dpsi).integral())
        beta = 0.5
        dpsip = dpsi_ini
        #dpsim = dpsip
        iter = 0
        diffp = 100
        #diffm = 100
        eps = 1.0e-6
        itermax = 100
        drho = calc_drho(self.psi0, dpsip, self.N)
        print("drho:", drho.integral())
        #while((diffp > 1e-5 or diffm > 1e-5) and iter<100):
        while((diffp > eps) and iter < itermax):
            iter += 1
            dpsip_old = dpsip
            #dpsim_old = dpsim
            #dv = self.calc_dv(dpsip) + self.calc_dv(dpsim)
            dv = self.calc_dv(dpsip_old)
            print((self.rho0 * dv).integral())
            dpsip, info = self.linearSolver(dpsip_old, dv, omega, eta)
            if info:
                print("Convergence not reached!")
                break
            drho = calc_drho(self.psi0, dpsip, self.N)
            print("drho:", drho.integral())
            norm = dpsip.norm()
            dpsip = dpsip.normalize(0.01)
            #if norm > 10:
            #    dpsip = dpsip.normalize()
            dpsip = beta * dpsip + (1.0 - beta) * dpsip_old
            diffp = (np.abs(dpsip - dpsip_old).integral())
            #dpsim, info = self.linearSolver(dpsim_old, dv, -omega, eta)
            #if info:
            #    break
            #dpsim = dpsim.normalize()
            #dpsim = beta * dpsim + (1.0 - beta) * dpsim_old
            #diffm = (np.abs(dpsim - dpsim_old).integral())

            #print("iter: {0:d}  diff: {1:f}, {2:f}".format(iter, diffp, diffm))
            print("iter: {0:d}  diff: {1:f} norm: {2:f}".format(iter, diffp, norm))

        if iter == itermax and diffp > eps:
            print("Convergence not reached!")
        return dpsip


    def __call__(self, dpsi_ini, omega_list, eta):

        dpsi_ini *= 0.01
        print(self.N * (self.psi0 * self.hamiltonian(self.psi0)).integral())
        f = np.zeros(np.shape(omega_list))
        for i, omega in enumerate(omega_list):
            print("omega: ", omega)
            dpsi = self.scf(dpsi_ini, omega, eta)
            psi1 = self.psi0 + dpsi
            for direc in range(3):
                mu = (np.conj(psi1) * self.grid.r[direc] * self.psi0).integral()
                f[i] += np.real(np.conj(mu) * mu)
            f[i] = f[i] * 2.0 / 3.0 * omega_list[i]
            dpsi_ini = dpsi

        return f


