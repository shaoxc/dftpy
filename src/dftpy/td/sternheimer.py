import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as linalg
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.utils.utils import calc_drho
from dftpy.td.operator import Operator
from typing import Tuple
from dftpy.linear_solver import cg, bicgstab
from dftpy.optimize import Dynamics
from dftpy.system import System
from dftpy.mpi.utils import sprint


class SternheimerOperator(Operator):

    def __init__(self, hamiltonian: Hamiltonian, e0: float, omega: float, eta: float) -> None:
        super(SternheimerOperator, self).__init__(hamiltonian.grid)
        self.hamiltonian = hamiltonian
        self.e0 = e0
        self.omega = omega
        self.eta = eta
        # sprint('omega: ', self.omega)

    @property
    def hamiltonian(self) -> Hamiltonian:
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: Hamiltonian) -> None:
        self.grid = hamiltonian.grid
        self._hamiltonian = hamiltonian

    def __call__(self, dpsi: DirectField) -> DirectField:
        return self.hamiltonian(dpsi) + (-self.e0 + self.omega + 1j * self.eta) * dpsi


class SternheimerEquationSolver(object):

    def __init__(self, hamiltonian: Hamiltonian, e0: float, psi0: DirectField, dv: DirectField, omega: float, eta: float) -> None:
        self.hamiltonian = hamiltonian
        self.e0 = e0
        self.psi0 = psi0
        self.dv = dv
        self.omega = omega
        self.eta = eta
        self.maxiter = 1000
        self.tol = 1e-7
        self.dpsi_ini = np.ones_like(psi0)
        self.dpsi_ini = self.dpsi_ini - (self.psi0 * self.dpsi_ini).integral() * self.psi0

    def _calc_b(self, dv: DirectField) -> DirectField:
        return -dv * self.psi0 + (self.psi0 * dv * self.psi0).integral() * self.psi0

    def linear_solver(self, omega: float, eta: float) -> Tuple[DirectField, int]:
        mat = SternheimerOperator(self.hamiltonian, self.e0, omega, eta)
        b = self._calc_b(self.dv)
        dpsi, status = bicgstab(A=mat, b=b, x0=self.psi0, tol=self.tol, maxiter=self.maxiter)
        if status != 0:
            raise Exception('Linear solver does not converge. Status: ', status)
        print('dpsi1: ', np.abs(b - mat(dpsi)).integral())
        # dpsi = dpsi - (self.psi0 * dpsi).integral() * self.psi0
        # print('dpsi2: ', np.abs(b - mat(dpsi)).integral())
        return dpsi, status

    def __call__(self) -> DirectField:
        dpsip, status = self.linear_solver(self.omega, self.eta)
        #dpsip += (self.psi0 * self.dv * self.psi0).integral() * self.psi0 / self.omega
        dpsim, status = self.linear_solver(-self.omega, self.eta)
        #dpsim += (self.psi0 * self.dv * self.psi0).integral() * self.psi0 / -self.omega
        drho = np.conj(self.psi0) * dpsip + self.psi0 * np.conj(dpsim)
        # drho = 2 * np.real(self.psi0 * dpsip)
        # drho2 = 2 * drho
        # drho2[1::, 1::, 1::] = drho[1::, 1::, 1::] - drho[-1:0:-1, -1:0:-1, -1:0:-1]
        # drho2 /= 2
        return drho


class Sternheimer(Dynamics):

    def __init__(self, system: System, functionals: TotalFunctional, drho0: DirectField, omega: float, e0: float, e: float) -> None:
        Dynamics.__init__(self, system)
        self.rho0 = system.field
        print((self.rho0*(self.rho0.grid.r[0]-self.rho0.grid.lattice[0][0]/2)).integral())
        self.N = self.rho0.integral()
        self.psi0 = np.sqrt(self.rho0 / self.N)
        self.psi0.complex = True
        self.functionals = functionals
        self.hamiltonian = Hamiltonian(v=self.functionals(self.rho0, calcType=['V']).potential)
        self.kxc = functionals.Subset(['KineticEnergyFunctional', 'XCFunctional'])
        #self.e0 = (self.psi0 * self.hamiltonian(self.psi0)).integral()
        self.e0 = e0
        self.drho0 = drho0
        self.drho = np.zeros_like(self.rho0)
        self.old_drho = drho0
        self.new_drho = np.zeros_like(self.rho0)
        self.omega = omega
        self.eta = 0.01
        self.max_steps = 1000
        self.tol = 1e-6
        self.diff = 0
        self.beta = 0.1
        self.attach(self.mixing)
        self._ftxc = None
        self.e = e

        dv = self.calc_dv(self.drho)
        self.sternheimer_equation_solver = SternheimerEquationSolver(hamiltonian=self.hamiltonian, e0=self.e0, psi0=self.psi0, dv=dv, omega=self.omega, eta=0)
        self.drho = self.sternheimer_equation_solver() * self.N
        sprint("drho: ", self.drho.integral())
        # from dftpy.formats.xsf import XSF
        # xsf = XSF(filexsf='drho3.xsf')
        # xsf.write(system=self.system, field=self.drho)
        # xsf = XSF(filexsf='dv.xsf')
        # xsf.write(system=self.system, field=dv)
        # dv = self.calc_dv(self.drho0)
        # self.sternheimer_equation_solver.dv = dv
        # self.sternheimer_equation_solver.eta = self.eta
        # self.p = self.drho0.copy()
        # self.r = self.drho0.copy()
        # self.ap = self.drho0 - self.sternheimer_equation_solver() * self.N
        # print(np.abs(self.ap).integral())

    def calc_dv(self, drho: DirectField) -> DirectGrid:
        if self._ftxc is None:
            self._ftxc = self.functionals(self.rho0, calcType=['V2']).v2rho2
        dvtxc = self._ftxc * drho
        hartree = Functional(type='HARTREE')
        dvh = hartree(drho, calcType=['V']).potential

        return dvtxc + dvh + (drho.grid.r[0] - drho.grid.lattice[0][0] / 2) * self.e *(1 - np.exp(-2 * np.abs(np.abs(drho.grid.r[0] - drho.grid.lattice[0][0] / 2) - drho.grid.lattice[0][0] / 2)))

    def step(self):
        self.old_drho = self.drho
        dv = self.calc_dv(drho = self.drho)
        self.sternheimer_equation_solver.dv = dv
        self.new_drho = self.sternheimer_equation_solver() * self.N
        sprint("drho: ", self.new_drho.integral())
        # self.old_drho = self.drho
        # alpha = (self.r * self.ap).integral() / (self.ap * self.ap).integral()
        # self.drho = self.drho + alpha * self.p
        # self.r = self.r - alpha * self.ap
        # dv = self.calc_dv(self.r)
        # self.sternheimer_equation_solver.dv = dv
        # ddrho = self.r - self.sternheimer_equation_solver() * self.N
        # beta = (ddrho * self.ap).integral() / (self.ap * self.ap).integral()
        # self.p = self.r + beta * self.p
        # self.ap = ddrho + beta * self.ap

    def converged(self):
        # self.diff = np.abs(self.drho - self.old_drho).integral()
        self.diff = np.abs(self.new_drho - self.old_drho).integral()
        return self.diff < self.tol

    def log(self):
        sprint("iter: {0:d}  diff: {1:.10e}".format(self.nsteps, self.diff))
        # sprint('p: ', np.abs(self.p).integral(), 'r: ',  np.abs(self.r).integral(), 'Ap: ', np.abs(self.ap).integral())

    def mixing(self):
        self.drho = self.beta * self.new_drho + (1.0 - self.beta) * self.old_drho

    # def scf(self, dpsi_ini, omega, eta):
    #     #print((psi1*psi1).integral())
    #     #dpsi = psi1 - self.psi*(self.psi*psi1).integral()
    #     #print((dpsi*dpsi).integral())
    #     beta = 0.5
    #     dpsip = dpsi_ini
    #     #dpsim = dpsip
    #     iter = 0
    #     diffp = 100
    #     #diffm = 100
    #     eps = 1.0e-6
    #     itermax = 100
    #     drho = calc_drho(self.psi0, dpsip, self.N)
    #     print("drho:", drho.integral())
    #     #while((diffp > 1e-5 or diffm > 1e-5) and iter<100):
    #     while((diffp > eps) and iter < itermax):
    #         iter += 1
    #         dpsip_old = dpsip
    #         #dpsim_old = dpsim
    #         #dv = self.calc_dv(dpsip) + self.calc_dv(dpsim)
    #         dv = self.calc_dv(dpsip_old)
    #         print((self.rho0 * dv).integral())
    #         dpsip, info = self.linearSolver(dpsip_old, dv, omega, eta)
    #         if info:
    #             print("Convergence not reached!")
    #             break
    #         drho = calc_drho(self.psi0, dpsip, self.N)
    #         print("drho:", drho.integral())
    #         norm = dpsip.norm()
    #         dpsip = dpsip.normalize(0.01)
    #         #if norm > 10:
    #         #    dpsip = dpsip.normalize()
    #         dpsip = beta * dpsip + (1.0 - beta) * dpsip_old
    #         diffp = (np.abs(dpsip - dpsip_old).integral())
    #         #dpsim, info = self.linearSolver(dpsim_old, dv, -omega, eta)
    #         #if info:
    #         #    break
    #         #dpsim = dpsim.normalize()
    #         #dpsim = beta * dpsim + (1.0 - beta) * dpsim_old
    #         #diffm = (np.abs(dpsim - dpsim_old).integral())
    #
    #         #print("iter: {0:d}  diff: {1:f}, {2:f}".format(iter, diffp, diffm))
    #         print("iter: {0:d}  diff: {1:f} norm: {2:f}".format(iter, diffp, norm))
    #
    #     if iter == itermax and diffp > eps:
    #         print("Convergence not reached!")
    #     return dpsip


    # def __call__(self, dpsi_ini, omega_list, eta):
    #
    #     dpsi_ini *= 0.01
    #     print(self.N * (self.psi0 * self.hamiltonian(self.psi0)).integral())
    #     f = np.zeros(np.shape(omega_list))
    #     for i, omega in enumerate(omega_list):
    #         print("omega: ", omega)
    #         dpsi = self.scf(dpsi_ini, omega, eta)
    #         psi1 = self.psi0 + dpsi
    #         for direc in range(3):
    #             mu = (np.conj(psi1) * self.grid.r[direc] * self.psi0).integral()
    #             f[i] += np.real(np.conj(mu) * mu)
    #         f[i] = f[i] * 2.0 / 3.0 * omega_list[i]
    #         dpsi_ini = dpsi
    #
    #     return f


