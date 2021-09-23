import numpy as np

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.linear_solver import _get_atol
from dftpy.mpi import sprint
from dftpy.optimize import Dynamics
from dftpy.utils.utils import calc_rho, calc_j


class PredictorCorrector(Dynamics):

    def __init__(self,
                 system,
                 propagator=None,
                 tol=None,
                 atol=None,
                 max_steps=None,
                 propagate_vector_potential=False,
                 int_t=None,
                 functionals=None,
                 Omega=None,
                 A_t=None,
                 A_tm1=None,
                 N0=None):
        Dynamics.__init__(self, system=system)
        self.propagator = propagator
        self.max_steps = max_steps
        self.tol = tol
        self.atol = atol
        self.propagate_vector_potential = propagate_vector_potential
        self.int_t = int_t
        self.functionals = functionals
        self.psi = self.system.field
        self.rho = calc_rho(self.psi)
        self.j = calc_j(self.psi)
        self.atol_rho = _get_atol(tol, atol, self.rho.norm())
        self.atol_j = _get_atol(tol, atol, np.max(self.j.norm()))
        self.old_rho_pred = np.zeros_like(self.rho)
        self.rho_pred = self.rho
        self.old_j_pred = np.zeros_like(self.j)
        self.j_pred = self.j
        self.Omega = Omega
        self.A_t = A_t
        self.A_tm1 = A_tm1
        self.N0 = N0
        self.psi_pred = None
        self.A_t_pred = None
        self.rho_corr = None
        self.j_corr = None
        self.A_t_corr = None
        self.diff_rho = None
        self.diff_j = None

    def step(self):
        if self.nsteps > 0:
            self.old_rho_pred = self.rho_pred
            self.old_j_pred = self.j_pred

        self.psi_pred, info = self.propagator(self.psi, self.int_t)
        self.rho_pred = calc_rho(self.psi_pred)
        self.j_pred = calc_j(self.psi_pred)
        if self.propagate_vector_potential:
            self.A_t_pred = np.empty_like(self.A_t)
            self.A_t_pred[:] = np.real(
                2 * self.A_t - self.A_tm1 - 4 * np.pi * self.N0 * self.A_t / self.Omega * self.int_t * self.int_t - 4.0 * np.pi * SPEED_OF_LIGHT * self.N0 / self.Omega * self.psi_pred.para_current(
                    sigma=0.025) * self.int_t * self.int_t)

        self.rho_corr = (self.rho + self.rho_pred) * 0.5
        self.j_corr = (self.j + self.j_pred) * 0.5
        func = self.functionals(self.rho_corr, calcType=["V"], current=self.j_corr)
        self.propagator.hamiltonian.v = func.potential
        if self.propagate_vector_potential:
            self.A_t_corr = (self.A_t + self.A_t_pred) * 0.5
            self.propagator.hamiltonian.A = self.A_t_corr

    def converged(self, *args):
        self.diff_rho = (self.old_rho_pred - self.rho_pred).norm()
        self.diff_j = np.max((self.old_j_pred - self.j_pred).norm())
        return self.diff_rho < self.atol_rho and self.diff_j < self.atol_j

    def print_not_converged_info(self):
        if self.max_steps > 1:
            sprint('Convergence not reached for Predictor-corrector')
            if self.diff_rho >= self.atol_rho:
                sprint('Diff in rho: {0:10.2e} > {1:10.2e}'.format(self.diff_rho, self.atol_rho))
            if self.diff_j >= self.atol_j:
                sprint('Diff in j: {0:10.2e} > {1:10.2e}'.format(self.diff_j, self.atol_j))
