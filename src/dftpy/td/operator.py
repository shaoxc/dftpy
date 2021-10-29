from dftpy.field import BaseField, DirectField, ReciprocalField
from dftpy.grid import BaseGrid, ReciprocalGrid
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Union
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from dftpy.time_data import timer
from dftpy.eigen_solver import power_iter, minimization, minimization2


class Operator(ABC):
    """
    Abstract class for operators which act on a DirectField or a ReciprocalField
    """
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, psi: BaseField, *args, **kwargs) -> BaseField:
        """
        Performs the operation on a wavefunction, needs to be implemented by subclasses

        Parameters
        ----------
        psi: the wavefunction the operator acts on

        Returns
        -------
        result: the result function in the same type as psi

        """
        pass

    def scipy_matvec_utils(self, grid: BaseGrid, *args, **kwargs) -> Callable:
        """
        Utility function that generates a matvec function that only takes psi.ravel() as input for SciPy

        Parameters
        ----------
        grid: the grid of the wavefunction psi
        reciprocal: whether the input field is a DirectField or a ReciprocalField

        Returns
        -------
        _scipy_matvec: the function that can be passed as the matvec function for SciPy LinearOperator.

        """
        def _scipy_matvec(psi_: np.ndarray) -> np.ndarray:
            if isinstance(grid, ReciprocalGrid):
                psi = ReciprocalField(grid, rank=1, griddata_3d=np.reshape(psi_, grid.nr))
            else:
                psi = DirectField(grid, rank=1, griddata_3d=np.reshape(psi_, grid.nr))
            prod = self(psi, *args, **kwargs)
            return prod.ravel()

        return _scipy_matvec

    def matvec_utils(self, *args, **kwargs) -> Callable:
        """
        Utility function that generates a matvec function that only takes psi as input

        Returns
        -------
        _matvec: the function only takes psi as input and does the operation on psi

        """

        def _matvec(psi: BaseField) -> BaseField:
            return self(psi, *args, **kwargs)

        return _matvec

    @timer('Diagonalize')
    def diagonalize(self, grid, numeig: int, return_eigenvectors: bool = True, scipy: bool = True,
                    x0: Union[float, None] = None, **kwargs) -> Tuple:
        if scipy:
            size = grid.nnr
            reciprocal = isinstance(grid, ReciprocalGrid)
            if reciprocal:
                dtype = np.complex128
            else:
                dtype = np.float64

            A = LinearOperator((size, size), dtype=dtype, matvec=self.scipy_matvec_utils(grid, **kwargs))

            if return_eigenvectors:
                eigenvalue_list, psis = eigsh(A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors)
                psi_list = []
                for i in range(numeig):
                    if reciprocal:
                        psi = ReciprocalField(grid, rank=1, griddata_3d=np.reshape(psis[:, i], grid.nr))
                    else:
                        psi = DirectField(grid, rank=1, griddata_3d=np.reshape(psis[:, i], grid.nr))
                    psi = psi.normalize()
                    psi_list.append(psi)
                return eigenvalue_list, psi_list
            else:
                eigenvalue_list, psis = eigsh(A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors)
                return eigenvalue_list,

        else:
            # mu, psi = power_iter(self.matvec_utils(**kwargs), x0)
            mu, psi = minimization2(self.matvec_utils(**kwargs), x0)
            return [mu], [psi]
