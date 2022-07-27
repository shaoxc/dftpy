import numpy as np
from numpy.typing import ArrayLike


class BaseCell(object):
    """
    Definition of the lattice of a system.

    Attributes
    ----------
    units: {'Bohr', 'Angstrom', 'nm', 'm'}, optional
        length units of the lattice vectors.
    lattice: array_like[3,3]
        matrix containing the lattice vectors of the cell (as its colums)
    omega: float
        volume of the cell in units**3

    """

    def __init__(self, lattice: ArrayLike, origin: ArrayLike = np.array([0.0, 0.0, 0.0]),
                 periodic: ArrayLike = np.array([True, True, True]), **kwargs):
        """

        Parameters
        ----------
        lattice: array_like(3, 3)
            matrix containing the direct/reciprocal lattice vectors (as its columns)
        origin: The coordinate of the origin of the cell
        units: {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            lattice is always passed as Bohr, but we can save a preferred unit for print purposes
        periodic: array_like(3, )
            vector containing whether each dimension is periodic

        """
        # lattice is always stored in atomic units: Bohr for direct lattices, 1/Bohr for reciprocal lattices
        self._lattice = np.asarray(lattice)
        self._origin = np.asarray(origin)
        self._volume = np.abs(np.dot(lattice[:, 0], np.cross(lattice[:, 1], lattice[:, 2])))
        self._periodic = np.asarray(periodic)
        self._lat_paras = np.diagonal(np.dot(self.lattice.T, self.lattice))

    def __eq__(self, other: 'BaseCell') -> bool:
        """
        Implement the == operator in the Cell class.

        The method is general and works even if the two cells use different
        length units.

        Parameters
        ----------
        other : Cell
            another cell object we are comparing to

        Returns
        -------
        out : Bool

        """
        if self is other:
            # if they refer to the same object, just cut to True
            return True

        for i_lat in range(3):
            lat0 = self.lattice[:, i_lat]
            lat1 = other.lattice[:, i_lat]
            if not np.isclose(lat0, lat1).all():
                return False

        return True

    @property
    def lattice(self):
        return self._lattice

    @property
    def origin(self):
        return self._origin

    @property
    def volume(self):
        return self._volume

    @property
    def periodic(self):
        return self._periodic

    @property
    def lat_paras(self):
        return self._lat_paras


class DirectCell(BaseCell):

    def __eq__(self, other: 'DirectCell') -> bool:
        """
        Implement the == operator in the DirectCell class.
        Refer to the __eq__ method of Cell for more information.
        """
        if not isinstance(other, DirectCell):
            raise TypeError("You can only compare a DirectCell with another DirectCell")
        return super().__eq__(other)

    def get_reciprocal(self, scale: ArrayLike = np.array([1.0, 1.0, 1.0]),
                       convention: str = "physics") -> 'ReciprocalCell':
        """
            Returns a new ReciprocalCell, the reciprocal cell of self
            The ReciprocalCell is scaled properly to include
            the scaled (*self.nr) reciprocal grid points
            -----------------------------
            Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
            physics convention defines the reciprocal lattice to be
            exp^{i G \cdot R} = 1
            Now we have the following "crystallographer's" definition ('crystallograph')
            which comes from defining the reciprocal lattice to be
            e^{2\pi i G \cdot R} =1
            In this case bg^T = at^{-1}
            -----------------------------
            Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        scale = np.asarray(scale)
        fac = 1.0
        if convention == "physics" or convention == "p":
            fac = 2 * np.pi
        bg = fac * np.linalg.inv(self.lattice)
        bg = bg.T
        reciprocal_lat = np.einsum("ij,j->ij", bg, scale)

        return ReciprocalCell(lattice=reciprocal_lat)


class ReciprocalCell(BaseCell):

    def __eq__(self, other: 'ReciprocalCell') -> bool:
        """
        Implement the == operator in the ReciprocalCell class.
        Refer to the __eq__ method of Cell for more information.
        """
        if not isinstance(other, ReciprocalCell):
            raise TypeError("You can only compare a ReciprocalCell with another ReciprocalCell")
        return super().__eq__(other)

    def get_direct(self, scale: ArrayLike = np.array([1.0, 1.0, 1.0]), convention: str = "physics") -> DirectCell:
        """
            Returns a new DirectCell, the direct cell of self
            The DirectCell is scaled properly to include
            the scaled (*self.nr) reciprocal grid points
            -----------------------------
            Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
            physics convention defines the reciprocal lattice to be
            exp^{i G \cdot R} = 1
            Now we have the following "crystallographer's" definition ('crystallograph')
            which comes from defining the reciprocal lattice to be
            e^{2\pi i G \cdot R} =1
            In this case bg^T = at^{-1}
            -----------------------------
            Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        scale = np.array(scale)
        fac = 1.0
        if convention == "physics" or convention == "p":
            fac = 1.0 / (2 * np.pi)
        at = np.linalg.inv(self.lattice.T * fac)
        direct_lat = np.einsum("ij,j->ij", at, 1.0 / scale)

        return DirectCell(lattice=direct_lat, origin=[0.0, 0.0, 0.0])

# def getrMIC(atm2, atm1, cell):
#     ds12 = atm1.spos - atm2.spos
#     for i in range(3):
#         ds12[i] = ds12[i] - np.round(ds12[i])
#         dr12 = s2r(ds12, cell)
#     return dr12
