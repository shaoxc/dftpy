from abc import ABC, abstractmethod
from typing import Union

from dftpy.field import DirectField, ReciprocalField
from dftpy.td.hamiltonian import Hamiltonian


class AbstractPropagator(ABC):
    """
    Abstract class for real-time propagators
    """

    def __init__(self, hamiltonian: Hamiltonian, interval: float):
        """

        Parameters
        ----------
        hamiltonian: Hamiltonian
            the time-dependent Hamiltonian
        interval: float
            the time interval for one time step

        """

        if isinstance(hamiltonian, Hamiltonian):
            self._hamiltonian = hamiltonian
        else:
            raise TypeError("hamiltonian must be a DFTpy Hamiltonian.")
        self._interval = interval

    @property
    def hamiltonian(self) -> Hamiltonian:
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: Hamiltonian) -> None:
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
