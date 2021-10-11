from abc import abstractmethod

from dftpy.field import BaseField
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.operator import Operator


class AbstractPropagator(Operator):
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
    def __call__(self, psi0: BaseField, **kwargs):
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
