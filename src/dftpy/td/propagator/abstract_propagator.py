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
        hamiltonian: the time-dependent Hamiltonian
        interval: the time interval for one time step

        """

        if isinstance(hamiltonian, Hamiltonian):
            self.hamiltonian = hamiltonian
        else:
            raise TypeError("hamiltonian must be a DFTpy Hamiltonian.")
        self.interval = interval

    @abstractmethod
    def __call__(self, psi0: Union[DirectField, ReciprocalField]):
        """
        Abstract method that performs one step of propagation. Should be implemented in child classes.
        Parameters
        ----------
        psi0: the initial wavefunction.

        Returns
        -------
        A tuple (psi1, status)
        psi1: Union[DirectField, ReciprocalField], the final wavefunction.
        status: int, 0: no issue, other numbers: has issues

        """
        pass
