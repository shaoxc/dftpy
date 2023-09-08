import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Tuple

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField, ReciprocalField, BaseField
from dftpy.grid import DirectGrid
from dftpy.time_data import timer
from dftpy.td.operator import Operator
from dftpy.optimization import Optimization
from dftpy.functional.functional_output import FunctionalOutput


class Hamiltonian(Operator):
    """
    Hamiltonian: 
    .. math::
        \hat{H} = \frac{1}{2}\left(-i\nabla-\frac{\mathbf{A}}{c}\right)^2 + v
    """

    def __init__(self, v=None, A=None, full=True):
        """

        Parameters
        ----------
        v: DirectField
            Effective scalar potential
        A: np.ndarray
            Vector potential which is constant in space
        full: bool
            Must be True if complex numbers are involved in the computation

        """
        self.full = full
        self.v = v
        self.A = A

    @property
    def v(self):
        return self._v

    @property
    def A(self):
        return self._A

    @v.setter
    def v(self, v):
        if isinstance(v, DirectField):
            self._v = v
            if v.grid.full == self.full:
                self.grid = v.grid
            else:
                self.grid = DirectGrid(lattice=v.grid.lattice, nr=v.grid.nr, origin=v.grid.origin, full=self.full, mp=v.grid.mp)
        elif v is None:
            self._v = None
            self.grid = None
        else:
            raise TypeError("v must be a DFTpy DirectField.")

    @A.setter
    def A(self, new_A):
        if new_A is None:
            self._A = None
        else:
            if np.size(new_A) != 3:
                raise AttributeError('Size of the A must be 3.')
            self._A = np.asarray(new_A)

    def __call__(self, psi, force_real=None, sigma=0.025, **kwargs):
        """
        Performs the Hamiltonian operation on psi.

        Parameters
        ----------
        psi: DirectField or ReciprocalField
            The wavefunction
        force_real: None or bool
            Determines the force_real flag for inverse FFT. If set to None, the flag will be determined base on
            whether psi is real or complex.
        sigma: float
            The smearing factor for gradient.

        Returns
        -------
        result: DirectField or ReciprocalField
        The result

        """
        return self.kinetic_operator(psi, force_real=force_real, sigma=sigma, **kwargs) + self.potential_operator(psi)

    def kinetic_operator(self, psi: BaseField, force_real=None, sigma=0.025, kpoint=np.array([0, 0, 0]), **kwargs) -> BaseField:
        kpoint = np.asarray(kpoint)
        reciprocal = isinstance(psi, ReciprocalField)
        if not reciprocal:
            if force_real is None:
                if np.isrealobj(psi):
                    force_real = True
                else:
                    force_real = False
            psi = psi.fft()

        if self._A is None:
            k_minus_a = kpoint
        else :
            k_minus_a = kpoint - self._A / SPEED_OF_LIGHT
        g = k_minus_a[:,None,None,None] + psi.grid.g
        gg = np.einsum("lijk,lijk->ijk", g, g)
        result = 0.5 * gg * psi
        if sigma :
            result *= np.exp(-psi.grid.gg * (sigma * sigma / 4.0))
        if not reciprocal:
            result = result.ifft(force_real=force_real)

        return result

    def potential_operator(self, psi: BaseField) -> BaseField:
        if isinstance(psi, ReciprocalField):
            return (self.v * psi.ifft()).fft()
        else:
            return self.v * psi

    @timer('Diagonalize')
    def diagonalize(self, numeig: int = 1, return_eigenvectors: bool = True, reciprocal: bool = False, diagonalization = 'scipy', **kwargs) -> Tuple:
        """
        Diagonalize the Hamiltonian and returns the lowest eigenvalues and optionally eigenvectors

        Parameters
        ----------
        numeig: int
            Number of eigenvalues to return.
        return_eigenvectors: bool
            Determine whether the eigenvectors will be returned.
        reciprocal: bool
            Determine the eigenvectors are calculated in real or reciprocal space.

        Returns
        -------
        Tuple (eigenvalue_list, ) or (eigenvalue_list, psi_list)
        eigenvalue_list: List[int]
            The list of eigenvalues
        psi_list: List[DirectField or ReciprocalField]
            The list of eigenvectors

        """
        if diagonalization == 'dftpy' :
            if numeig > 1: raise AttributeError(f"{diagonalization} only can compute one eigenvalue.")
            mu, x = self.diagonalize_optimization(reciprocal=reciprocal, **kwargs)
            return [mu], [x]

        if reciprocal:
            reci_grid = self.grid.get_reciprocal()
            size = reci_grid.nnr
            dtype = np.complex128
        else:
            size = self.grid.nnr
            dtype = np.float64

        A = LinearOperator((size, size), dtype=dtype, matvec=self.scipy_matvec_utils(reciprocal=reciprocal, **kwargs))
        eigenvalue_list, psis = eigsh(A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors)

        if return_eigenvectors:
            psi_list = []
            for i in range(numeig):
                if reciprocal:
                    psi = ReciprocalField(reci_grid, rank=1, griddata_3d=np.reshape(psis[:, i], reci_grid.nr))
                else:
                    psi = DirectField(self.grid, rank=1, griddata_3d=np.reshape(psis[:, i], self.grid.nr))
                psi = psi / np.sqrt((np.real(psi) * np.real(psi) + np.imag(psi) * np.imag(psi)).integral())
                psi_list.append(psi)
            psis = psi_list

        return eigenvalue_list, psis

    def energy(self, psi, hpsi=None, **kwargs):
        """
        Calculate the expectation value of Hamiltonian on wavefunction psi

        Parameters
        ----------
        psi: DirectField or ReciprocalField
            The wavefunction

        Returns
        -------
        energy: float

        """
        if hpsi is None: hpsi = self(psi, **kwargs)
        if np.isrealobj(psi):
            value = (psi * hpsi).integral()
        else:
            value = (np.conj(psi) * hpsi).real.integral()
        return value

    def diagonalize_optimization(self, x0=None, tol: float = 1.0e-13, maxiter: int = 10000, **kwargs):
        evaluator = self.compute_utils(calcType=None, phi=None, lphi=True, **kwargs)
        optimizer = Optimization(EnergyEvaluator=evaluator, optimization_options={'econv': tol, 'maxiter': maxiter})
        optimizer(guess_rho=x0*x0, guess_phi=x0, lphi=True)

        x = optimizer.phi.normalize()
        mu = optimizer.mu

        return mu, x

    def compute_utils(self, **kwargs):
        def compute(rho, calcType=None, phi=None, lphi=True, **options):
            result = FunctionalOutput(name='Hpsi')
            hphi = self(phi, **kwargs)
            if 'E' in calcType:
                result.energy = self.energy(phi, hphi)
            if 'V' in calcType:
                result.potential = hphi/phi
            return result

        return compute
