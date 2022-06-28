from abc import ABC, abstractmethod
from typing import Union

from dftpy.field import DirectField, ReciprocalField
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.operator import Operator


class AbstractPropagator(ABC):
    """
    Abstract class for real-time propagators
    """

    def __init__(self, hamiltonian: Operator, interval: float):
        """

        Parameters
        ----------
        hamiltonian: Operator
            the time-dependent Hamiltonian
        interval: float
            the time interval for one time step

        """

        if isinstance(hamiltonian, Operator):
            self._hamiltonian = hamiltonian
        else:
            raise TypeError("hamiltonian must be a DFTpy Operator.")
        self._interval = interval

    @property
    def hamiltonian(self) -> Operator:
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: Operator) -> None:
        self._hamiltonian = hamiltonian

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, interval: float) -> None:
        self._interval = interval

    @abstractmethod
    def __call__(self, psi0: Union[DirectField, ReciprocalField]):
        """
        Abstract method that performs one step of propagation. Should be implemented in child classes.
        Parameters
        ----------
        psi0: DirectField or ReciprocalField
            the initial wavefunction.

        Returns
        -------
        A tuple (psi1, status)
        psi1: DirectField or ReciprocalField, same as psi0
            The final wavefunction.
        status: int
            0: no issue, other numbers: has issues

        """
        pass
