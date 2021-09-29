import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField, ReciprocalField
from dftpy.time_data import timer
from dftpy.eigen_solver import power_iter, minimization, minimization2


class Hamiltonian(object):

    def __init__(self, v=None, A=None):
        self.v = v
        self.A = A
        if self.v is None:
            self.grid = None
        else:
            self.grid = v.grid

    @property
    def v(self):
        return self._v

    @property
    def A(self):
        return self._A

    @v.setter
    def v(self, new_v):
        if isinstance(new_v, DirectField):
            self._v = new_v
            self.grid = new_v.grid
        elif new_v is None:
            self._v = None
            self.grid = None
        else:
            raise TypeError("v must be a DFTpy DirectField.")

    @A.setter
    def A(self, new_A):
        if new_A is None:
            self._A = None
        else:
            self._A = np.asarray(new_A)
            if np.size(new_A) != 3:
                raise AttributeError('Size of the A must be 3.')
            ones = np.ones(self.grid.nr)
            self.a_field = DirectField(self.grid, rank=3, griddata_3d=np.asarray(
                [self._A[0] * ones, self._A[1] * ones, self._A[2] * ones]) / SPEED_OF_LIGHT)

    def __call__(self, psi, force_real=None, sigma=0.025, k=None):
        if isinstance(psi, DirectField):
            if force_real is None:
                if np.isrealobj(psi):
                    force_real = True
                else:
                    force_real = False
            if self._A is None:
                if k is None or np.allclose(k, 0, atol=1e-10):
                    return -0.5 * psi.laplacian(force_real=force_real, sigma=sigma) + self.v * psi
                else:
                    ones = np.ones(self.grid.nr)
                    k_field = DirectField(self.grid, rank=3, griddata_3d=np.asarray(
                [k[0] * ones, k[1] * ones, k[2] * ones]))
                    #if sigma is None:
                    #    flag = 'standard'
                    #else:
                    #    flag = 'supersmooth'
                    #intermediate_result1 = -1.0j * psi.gradient(flag=flag, force_real=False, sigma=sigma) + k_field * psi
                    #intermediate_result2 = -1.0j * intermediate_result1.divergence(flag=flag, force_real=False, sigma=sigma) + k_field.dot(intermediate_result1)
                    #return 0.5 * intermediate_result2 + self.v * psi
                    psi_fft = psi.fft()
                    intermediate_result1 = 0.5 * psi_fft.grid.gg * psi_fft
                    intermediate_result1 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                    intermediate_result2 = - psi_fft.grid.g * psi_fft
                    intermediate_result2 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                    return intermediate_result1.ifft(force_real=force_real) - k_field.dot(
                        intermediate_result2.ifft(force_real=force_real)) + 0.5 * k_field.dot(
                        k_field) * psi + self.v * psi
            else:
                psi_fft = psi.fft()
                intermediate_result1 = 0.5 * psi_fft.grid.gg * psi_fft
                intermediate_result1 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                intermediate_result2 = - psi_fft.grid.g * psi_fft
                intermediate_result2 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                return intermediate_result1.ifft(force_real=force_real) + self.a_field.dot(
                    intermediate_result2.ifft(force_real=force_real)) + 0.5 * self.a_field.dot(
                    self.a_field) * psi + self.v * psi
                # return -0.5 * psi.laplacian(force_real = force_real, sigma=sigma) + 0.5 * self.a_field.dot(self.a_field) * psi + 1.0j * self.a_field.dot(psi.gradient(flag = "supersmooth", force_real = force_real, sigma=sigma)) + self.v * psi
                # return -0.5 * psi.laplacian(force_real = force_real, sigma=sigma) + 0.5 * self.a_field.dot(self.a_field) * psi + 0.5j * self.a_field.dot(psi.gradient(force_real = force_real)) + 0.5j * (self.a_field * psi).divergence(force_real=force_real) + self.v * psi

        elif isinstance(psi, ReciprocalField):
            return 0.5 * psi.grid.gg * psi + (self.v * psi.ifft()).fft
        else:
            raise TypeError("psi must be a DFTpy DirectField or ReciprocalField.")

    def matvecUtil(self, reciprocal=False, **kwargs):
        if reciprocal:
            reci_grid = self.grid.get_reciprocal()

        def matvec(psi_):
            if reciprocal:
                psi = ReciprocalField(reci_grid, rank=1, griddata_3d=np.reshape(psi_, reci_grid.nr))
            else:
                psi = DirectField(self.grid, rank=1, griddata_3d=np.reshape(psi_, self.grid.nr))
            prod = self(psi, **kwargs)
            return prod.ravel()

        return matvec

    def matvecUtil2(self, **kwargs):

        def matvec(psi):
            return self(psi, **kwargs)

        return matvec

    @timer('Diagonalize')
    def diagonalize(self, numeig, return_eigenvectors=True, reciprocal=False, scipy=True, x0=None, **kwargs):

        if scipy:
            if reciprocal:
                reci_grid = self.grid.get_reciprocal()
                size = reci_grid.nnr
                dtype = np.complex128
            else:
                size = self.grid.nnr
                dtype = np.float64

            A = LinearOperator((size, size), dtype=dtype, matvec=self.matvecUtil(reciprocal, **kwargs))

            if return_eigenvectors:
                Es, psis = eigsh(A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors)
                psi_list = []
                for i in range(numeig):
                    if reciprocal:
                        psi = ReciprocalField(reci_grid, rank=1, griddata_3d=np.reshape(psis[:, i], reci_grid.nr))
                    else:
                        psi = DirectField(self.grid, rank=1, griddata_3d=np.reshape(psis[:, i], self.grid.nr))
                    psi = psi / np.sqrt((np.real(psi) * np.real(psi) + np.imag(psi) * np.imag(psi)).integral())
                    psi_list.append(psi)
                return Es, psi_list
            else:
                Es, psis = eigsh(A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors)
                return Es

        else:
            A = self.matvecUtil2(**kwargs)
            #mu, psi = power_iter(A, x0)
            mu, psi = minimization2(A, x0)
            return [mu], [psi]

