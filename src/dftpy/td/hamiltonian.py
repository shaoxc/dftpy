import numpy as np
from numpy.typing import ArrayLike

from typing import Tuple, Union

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField, ReciprocalField, BaseField
from dftpy.grid import DirectGrid, ReciprocalGrid, BaseGrid

from dftpy.td.operator import Operator


class Hamiltonian(Operator):

    def __init__(self, potential: Union[DirectField, None] = None,
                 vector_potential: Union[ArrayLike, None] = None) -> None:
        self.potential = potential
        self.vector_potential = vector_potential

    @property
    def potential(self) -> DirectField:
        return self._v

    @property
    def vector_potential(self) -> np.ndarray:
        return self._A

    @potential.setter
    def potential(self, potential: Union[DirectField, None]) -> None:
        if isinstance(potential, DirectField):
            self._v = potential
        elif potential is None:
            self._v = None
        else:
            raise TypeError("v must be a DFTpy DirectField.")

    @vector_potential.setter
    def vector_potential(self, vector_potential: Union[ArrayLike, None]) -> None:
        if vector_potential is None:
            self._A = np.array([0, 0, 0])
        else:
            self._A = np.asarray(vector_potential)
            if np.size(vector_potential) != 3:
                raise AttributeError('Size of the A must be 3.')

    def __call__(self, psi, force_real=None, sigma=0.025, k_point=np.array([0, 0, 0])):
        return self.kinetic_operator(psi, force_real=force_real, sigma=sigma, k_point=k_point) + self.potential_operator(psi)

    def kinetic_operator(self, psi: BaseField, force_real=None, sigma=0.025, k_point=np.array([0, 0, 0])) -> BaseField:
        k_point = np.asarray(k_point)
        reciprocal = isinstance(psi, ReciprocalField)
        if not reciprocal:
            if force_real is None:
                if np.isrealobj(psi):
                    force_real = True
                else:
                    force_real = False
            psi = psi.fft()

        k_minus_a = k_point - self._A / SPEED_OF_LIGHT
        term1 = 0.5 * psi.grid.gg * psi
        term1 *= np.exp(-psi.grid.gg * sigma * sigma / 4.0)
        term2 = np.einsum('i,ijkl->jkl', k_minus_a, psi.grid.g) * psi
        term2 *= np.exp(-psi.grid.gg * sigma * sigma / 4.0)
        term3 = np.einsum('i,i->', k_minus_a, k_minus_a) * psi
        result = term1 + term2 + term3
        if not reciprocal:
            result = result.ifft(force_real=force_real)

        return result

    def potential_operator(self, psi: BaseField) -> BaseField:
        if isinstance(psi, ReciprocalField):
            return (self.potential * psi.ifft()).fft()
        else:
            return self.potential * psi
