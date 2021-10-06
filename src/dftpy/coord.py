import numpy as np
from numpy.typing import ArrayLike
from typing import Union

from dftpy.cell import BaseCell


class Coord(np.ndarray):
    """
    Array representing coordinates in real space under periodic boundary conditions.

    Attributes
    ----------
    _cell : BaseCell
        The unit cell associated to the coordinates.
    _basis : {'Cartesian', 'Crystal'}
        Describes whether the array contains crystal or cartesian coordinates.

    """

    cart_names = ["Cartesian", "Cart", "Ca", "R"]
    crys_names = ["Crystal", "Crys", "Cr", "S"]

    def __new__(cls, pos: ArrayLike, cell: BaseCell, basis: str = "Cartesian"):
        """
        Parameters
        ----------
        pos : array_like[..., 3]
            Array containing a single or a set of 3D coordinates.
        cell : DirectCell
            The unit cell to be associated to the coordinates.
        basis : {'Cartesian', 'Crystal'}
            matrix containing the direct lattice vectors (as its columns)

        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if not isinstance(cell, (BaseCell,)):
            raise TypeError(type(cell), 'is not BaseCell')

        if basis in Coord.cart_names:
            basis = Coord.cart_names[0]
        elif basis in Coord.crys_names:
            basis = Coord.crys_names[0]
        else:
            raise NameError("Unknown basis name: {}".format(basis))

        # Internally we always use Bohr, convert accordingly
        obj = np.asarray(pos, dtype=float).view(cls)

        # add the new attribute to the created instance
        obj._basis = basis
        obj._cell = cell
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self._cell = getattr(obj, "_cell", None)
        self._basis = getattr(obj, "_basis", None)
        # We do not need to return anything

    def __add__(self, other: ArrayLike) -> 'Coord':
        """
        Implement the '+' operator for the Coord class.

        Parameters
        ----------
        other : Coord | float | int | array_like
            What is to be summed to self.

        Returns
        -------
        out : Coord
            The sum of self and other.

        """
        if isinstance(other, type(self)):
            if self.cell == other.cell:
                other = other.to_basis(self.basis)
            else:
                raise Exception("Two Coord objects can only be added if they are represented in the same cell")

        return np.ndarray.__add__(self, other)

    def __mul__(self, scalar: Union[int, float]) -> 'Coord':
        """ Implement the scalar multiplication"""
        if isinstance(scalar, (int, float)):
            return np.multiply(self, scalar)
        else:
            raise TypeError("Coord can only be multiplied by a int or float scalar")

    @property
    def cell(self):
        return self._cell

    @property
    def basis(self):
        return self._basis

    def to_cart(self) -> 'Coord':
        """
        Converts the coordinates to Cartesian and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have basis='Cartesian'.
        """
        if self.basis in Coord.cart_names:
            return self
        else:
            pos = s2r(self, self.cell)
            return Coord(pos=pos, cell=self.cell, basis=Coord.cart_names[0])

    def to_crys(self) -> 'Coord':
        """
        Converts the coordinates to Crystal and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have basis='Crystal'.
        """
        if self.basis in Coord.crys_names:
            return self
        else:
            pos = r2s(self, self.cell)
            return Coord(pos=pos, cell=self.cell, basis=Coord.crys_names[0])

    def to_basis(self, basis) -> 'Coord':
        """
        Converts the coordinates to the desired basis and return a new object.

        Parameters
        ----------
        basis : {'Cartesian', 'Crystal'}
            basis to which the coordinates are converted.
        Returns
        -------
        out : Coord
            New Coord object insured to have basis=basis.
        """
        if basis in Coord.crys_names:
            return self.to_crys()
        elif basis in Coord.cart_names:
            return self.to_cart()
        else:
            raise NameError("Trying to convert to an unknown basis")

    def d_mic(self, other: 'Coord') -> 'Coord':
        """
        Calculate the vector connecting two Coord using the minimum image convention (MIC).

        Parameters
        ----------
        other : Coord

        Returns
        -------
        out : Coord
            shortest vector connecting self and other with the same basis as self.

        """
        ds12 = other.to_crys() - self.to_crys()
        ds12 -= np.round(ds12)
        return ds12.to_basis(self.basis)

    def dd_mic(self, other: 'Coord') -> float:
        """
        Calculate the distance between two Coord using the minimum image convention (MIC).

        Parameters
        ----------
        other : Coord

        Returns
        -------
        out : float
            the minimum distance between self and other from applying the MIC.

        """
        return self.d_mic(other).length()

    def length(self) -> float:
        """
        Calculate the length of a Coord array.

        Returns
        -------
        out : float
            The length of the Coord array, in the same units as self.cell.

        """
        return np.sqrt(np.dot(self.to_cart(), self.to_cart()))

    def change_of_basis(self, new_cell, new_origin=np.array([0.0, 0.0, 0.0])):
        """
        Perform a change of basis on the coordinates.

        Parameters
        ----------
        new_cell : Cell
            Cell representing the new coordinate system (i.e. the new basis)
        new_origin : array_like[3]
            Origin of the new coordinate system.

        Returns
        -------
        out : Coord
            Coord in the new basis.

        """
        # M = np.dot(self.cell.bg, new_cell.bg)
        # P = np.linalg.inv(M)
        # new_pos = np.dot(P, self.to_crys())
        # return Coord(new_pos, cell=new_cell)
        raise NotImplementedError("Generic change of basis non implemented yet in the Coord class")


def r2s(pos: ArrayLike, cell: BaseCell) -> np.ndarray:
    """
    Convert from crystal coordinates to Cartesian coordinates

    Parameters
    ----------
    pos: the crystal coordinates to convert
    cell: the unit cell pos is based to

    Returns
    -------
    xyzs: the Cartesian coordinates

    """
    # Vectorize the code: the right most axis is where the coordinates are
    pos = np.asarray(pos)
    bg = np.linalg.inv(cell.lattice)
    xyzs = np.einsum("...j,kj->...k", pos, bg)
    return xyzs


def s2r(pos: ArrayLike, cell: BaseCell) -> np.ndarray:
    """
    Convert from Cartesian coordinates to crystal coordinates

    Parameters
    ----------
    pos: the Cartesian coordinates to convert
    cell: the unit cell the result is based to

    Returns
    -------
    xyzr: the crystal coordinates

    """
    # Vectorize the code: the right most axis is where the coordinates are
    pos = np.asarray(pos)
    xyzr = np.einsum("...j,kj->...k", pos, cell.lattice)
    return xyzr
