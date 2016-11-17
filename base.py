"""
    pbcpy is a python package to seamlessly tackle periodic boundary conditions.

    Copyright (C) 2016 Alessandro Genova (ales.genova@gmail.com).

    pbcpy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    pbcpy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with pbcpy.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from .constants import LEN_CONV


class Cell(object):
    """
    Definition of the lattice of a system.

    Attributes
    ----------
    units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
        length units of the lattice vectors.
    at : array_like[3,3]
        matrix containing the direct lattice vectors (as its colums)
    bg : array_like[3,3]
        matrix containing the reciprocal lattice vectors (i.e. inverse of at)
    omega : float
        volume of the cell in units**3

    """
    def __init__(self, at, units='Bohr'):
        """
        Parameters
        ----------
        at : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.

        """
        self.at = np.asarray(at)
        self.units = units
        self.bg = np.linalg.inv(at)
        self.omega = np.dot(at[:, 0], np.cross(at[:, 1], at[:, 2]))
        # self.alat = np.sqrt(np.dot(at[:][0], at[:][0]))

    def __eq__(self, other):
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

        eps = 1e-4
        conv = LEN_CONV[other.units][self.units]

        for ilat in range(3):
            lat0 = self.at[:, ilat]
            lat1 = other.at[:, ilat] * conv
            overlap = np.dot(lat0, lat1) / np.dot(lat0, lat0)
            if abs(1 - overlap) > eps:
                return False

        return True

    def conv(self, units):
        """
        Convert the length units of the cell, and return a new object.

        Parameters
        ----------
        units : {'Bohr', 'Angstrom', 'nm', 'm'}
            The desired length units of the Cell in output.

        Returns
        -------
        out : Cell
            New cell object with changed length unit.
        """
        if self.units == units:
            return self
        else:
            return Cell(at=self.at*LEN_CONV[self.units][units], units=units)


class Coord(np.ndarray):
    """
    Array representing coordinates in periodic boundary conditions.

    Attributes
    ----------
    cell : Cell
        The unit cell associated to the coordinates.
    ctype : {'Cartesian', 'Crystal'}
        Describes whether the array contains crystal or cartesian coordinates.

    """
    cart_names = ['Cartesian', 'Cart', 'Ca', 'R']
    crys_names = ['Crystal', 'Crys', 'Cr', 'S']

    def __new__(cls, pos, cell=None, ctype='Cartesian', units='Bohr'):
        """
        Parameters
        ----------
        pos : array_like[..., 3]
            Array containing a single or a set of 3D coordinates.
        cell : Cell
            The unit cell to be associated to the coordinates.
        ctype : {'Cartesian', 'Crystal'}
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            If cell is missing, it specifies the units of the versors.
            Overridden by cell.units otherwise.

        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pos, dtype=float).view(cls)
        # add the new attribute to the created instance
        if cell is None:
            # If no cell in input, coordinates are purely cartesian,
            # i.e. the lattice vectors are three orthogonal versors i, j, k.
            obj.cell = Cell(np.identity(3), units=units)
        else:
            obj.cell = cell
        obj.ctype = ctype
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.cell = getattr(obj, 'cell', None)
        self.ctype = getattr(obj, 'ctype', None)
        # We do not need to return anything

    def __add__(self, other):
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
        # if isinstance(other, Coord):
            if self.cell == other.cell:
                other = other.conv(self.cell.units).to_ctype(self.ctype)
            else:
                return Exception

        return np.ndarray.__add__(self, other)

    def to_cart(self):
        """
        Converts the coordinates to Cartesian and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have ctype='Cartesian'.
        """
        if self.ctype in Coord.cart_names:
            return self
        else:
            pos = s2r(self, self.cell)
            return Coord(pos=pos, cell=self.cell, ctype=Coord.cart_names[0])

    def to_crys(self):
        """
        Converts the coordinates to Crystal and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have ctype='Crystal'.
        """
        if self.ctype in Coord.crys_names:
            return self
        else:
            pos = r2s(self, self.cell)
            return Coord(pos=pos, cell=self.cell, ctype=Coord.crys_names[0])

    def to_ctype(self, ctype):
        """
        Converts the coordinates to the desired ctype and return a new object.

        Parameters
        ----------
        ctype : {'Cartesian', 'Crystal'}
            ctype to which the coordinates are converted.
        Returns
        -------
        out : Coord
            New Coord object insured to have ctype=ctype.
        """
        if ctype in Coord.crys_names:
            return self.to_crys()
        elif ctype in Coord.cart_names:
            return self.to_cart()

    def d_mic(self, other):
        """
        Calculate the vector connecting two Coord using the minimum image convention (MIC).

        Parameters
        ----------
        other : Coord

        Returns
        -------
        out : Coord
            shortest vector connecting self and other with the same ctype as self.

        """
        ds12 = other.to_crys() - self.to_crys()
        for i in range(3):
            ds12[i] = ds12[i] - round(ds12[i])
        # dr12 = s2r(ds12, cell)
        return ds12.to_ctype(self.ctype)

    def dd_mic(self, other):
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
        return self.d_mic(other).lenght()

    def lenght(self):
        """
        Calculate the legth of a Coord array.

        Returns
        -------
        out : float
            The lenght of the Coord array, in the same units as self.cell.

        """
        return np.sqrt(np.dot(self.to_cart(), self.to_cart()))

    def conv(self, new_units):
        """
        Converts the units of the Coord array.

        Parameters
        ----------
        new_units : {'Bohr', 'Angstrom', 'nm', 'm'}

        Returns
        -------
        out : Coord

        """
        # new_at = self.cell.at.copy()
        new_at = self.cell.at.copy()
        new_at *= LEN_CONV[self.cell.units][new_units]
        # new_cell = Cell(new_at,units=new_units)
        return Coord(self.to_crys(), Cell(new_at, units=new_units),
                     ctype=Coord.crys_names[0]).to_ctype(self.ctype)

    def change_of_basis(self, new_cell, new_origin=np.array([0., 0., 0.])):
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
        M = np.dot(self.cell.bg, new_cell.bg)
        P = np.linalg.inv(M)
        new_pos = np.dot(P, self.to_crys())
        return Coord(new_pos, cell=new_cell)


class pbcarray(np.ndarray):

    def __new__(cls, pos):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pos).view(cls)
        # add the new attribute to the created instance
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        # self.cell = getattr(obj, 'cell', None)
        # self.ctype = getattr(obj, 'ctype', None)
        # We do not need to return anything

    def __getitem__(self, index):
        """
        All the possible slices with pbcarray

        """
        shape_ = self.shape
        rank = len(shape_)

        slices = self._reconstruct_full_slices(shape_, index)

        # slices = self._order_slices(shape_, slices)

        # Now actually slice with pbc along each direction.
        newarr = np.asarray(self)
        for idim, sli in slices:
            start = sli.start
            stop = sli.stop
            step = sli.step
            step_ = sli.step

            if step is None:
                step = 1

            if step > 0:
                if start is None:
                    start = 0
                if stop is None:
                    stop = shape_[idim]

            elif step < 0:
                if start is None:
                    start = shape_[idim]
                if stop is None:
                    stop = 0

            lower = min(start, stop)
            upper = max(start, stop)
            span = upper - lower

            # If the beginning of the slice does not coincide with a grid point
            # equivalent to 0, roll the array along that axis until it does
            roll = 0
            if lower % shape_[idim] != 0:
                roll = -lower % shape_[idim]
                newarr = np.roll(newarr, roll, axis=idim)

            # If the span of the slice extends beyond the boundaries of the array,
            # pad the array along that axis until we have enough elements.
            if span > shape_[idim]:
                pad_tup = [(0, 0)] * rank
                pad_tup[idim] = (0, span - shape_[idim])
                newarr = np.pad(newarr, pad_tup, mode='wrap')

            # And now get the slice of the array allong the axis.
            slice_tup = [slice(None)]*rank
            if step < 0:
                slice_tup[idim] = slice(
                    start + roll, stop + roll, step_)
            else:
                slice_tup[idim] = slice(None, span, step)

            slice_tup = tuple(slice_tup)
            newarr = newarr[slice_tup]

        return newarr

    def _reconstruct_full_slices(self, shape_, index):
        """
        Auxiliary function for __getitem__ to reconstruct the explicit slicing
        of the array if ellipsis and missing axes

        """
        if not isinstance(index, tuple):
            index = (index,)
        slices = []
        idx_len, rank = len(index), len(shape_)

        for slice_ in index:
            if slice_ is Ellipsis:
                slices.extend([slice(None)] * (rank+1-idx_len))
            elif isinstance(slice_, slice):
                slices.append(slice_)
            elif isinstance(slice_, (int)):
                slices.append(slice(slice_,slice_+1))

        sli_len = len(slices)
        if sli_len > rank:
            msg = 'too many indices for array'
            raise IndexError(msg)
        elif sli_len < rank:
            slices.extend([slice(None)]*(rank-sli_len))
            # Add info about the dimension the slice refers to so we can keep
            # track if we reorder them later.

        slices = list(zip(range(rank), slices))

        return slices

    def _order_slices(self, shape_, slices):
        """
        Order the slices span in ascending order.
        When we are slicing a pbcarray we might be rolling and padding the array
        so it's probably a good idea to make the array as small as possible
        early on.

        """
        pass
        # for idim, sli in zip(*zip(slices), :


def r2s(pos, cell):
    # Vectorize the code: the right most axis is where the coordinates are
    pos = np.asarray(pos)
    xyzs = np.tensordot(cell.bg, pos.T, axes=([-1], 0)).T
    # xyzs = np.dot(cell.bg, pos)
    return xyzs


def s2r(pos, cell):
    # Vectorize the code: the right most axis is where the coordinates are
    pos = np.asarray(pos)
    xyzr = np.tensordot(cell.at, pos.T, axes=([-1], 0)).T
    return xyzr


def getrMIC(atm2, atm1, cell):
    ds12 = atm1.spos - atm2.spos
    for i in range(3):
        ds12[i] = ds12[i] - round(ds12[i])
        dr12 = s2r(ds12, cell)
    return dr12
