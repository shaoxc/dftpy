import os
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from dftpy.base import Coord,DirectCell
from dftpy.field import ReciprocalField, DirectField
from dftpy.functional_output import Functional
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.ewald import CBspline


class Atom(object):
    def __init__(self, Z=None, Zval=None, label=None, pos=None, cell=None, basis="Cartesian"):
        """
        Atom class handles atomic position, atom type and local pseudo potentials.
        """

        if Zval is None:
            self.Zval = {}
        else:
            self.Zval = Zval
        # self.pos = Coord(pos, cell, basis='Cartesian')
        self._pos = Coord(pos, cell, basis=basis).to_cart()
        self.nat = len(pos)
        self.Z = Z
        self._ncharge = None

        # check label
        self.labels = []
        if label is not None:
            self.labels = []
            for i, item in enumerate(label):
                if str(item).isdigit():
                    self.labels.append(z2lab[int(item)])
                else :
                    self.labels.append(item)
            if self.Z is None:
                self.Z = []
                for item in self.labels:
                    self.Z.append(z2lab.index(item))
        else :
            for item in self.Z:
                self.labels.append(z2lab[item])

        self.labels = np.asarray(self.labels)
        self.Z = np.asarray(self.Z)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.nat = len(self._pos)
        if self._ncharge is not None :
            self._ncharge = self.get_ncharge()

    @property
    def ncharge(self):
        if self._ncharge is None :
            self._ncharge = self.get_ncharge()
        return self._ncharge

    def get_ncharge(self):
        if self.Zval is None:
            raise Exception("Must set 'Zval' first")
        ncharge = 0
        for item, n in zip(*np.unique(self.labels, return_counts=True)):
            ncharge += self.Zval[item] * n
        return ncharge

    def set_Zval(self, labels=None):
        if self.Zval is None:
            raise Exception("Must initialize Pseudo Potential with ReadPseudo")

    def strf(self, reciprocal_grid, iatom):
        """
        Returns the Structure Factor associated to i-th ion.
        """
        a = np.exp(-1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.pos[iatom]))
        return a

    def istrf(self, reciprocal_grid, iatom):
        a = np.exp(1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.pos[iatom]))
        return a

    def __getitem__(self, i):
        atoms = self.__class__(Z=self.Z[i].copy(), Zval=self.Zval, label=self.labels[i].copy(), pos=self.pos[i].copy(), cell = self.pos.cell, basis = self.pos.basis)
        return atoms

    def __delitem__(self, i):
        mask = np.ones_like(self.labels, dtype = bool)
        mask[i] = False
        self.labels = self.labels[mask]
        self.pos = self.pos[mask]
        self.Z = self.Z[mask]
        self.nat = len(self.pos)

    def __str__(self): 
        return '\n'.join(['%20s : %s' % item for item in self.__dict__.items()]) 

    def repeat(self, reps=1):
        reps = np.ones(3, dtype='int')*reps
        pos = self.pos.copy()
        Z = self.Z.copy()
        Zval = self.Zval.copy()
        lattice = self.pos.cell.lattice.copy()
        nat = self.nat
        rep = np.prod(reps)
        #-----------------------------------------------------------------------
        pos = np.tile(pos, (rep, 1))
        Z = np.tile(Z, rep)
        ia = 0
        ixyzA = np.mgrid[:reps[0],:reps[1],:reps[2]].reshape((3, -1))
        for i in range(rep):
            item = ixyzA[:, i]
            ib = ia + nat
            pos[ia:ib] += np.dot(lattice, item)
            ia = ib
        for i in range(3):
            lattice[:, i] *= reps[i]
        cell = DirectCell(lattice)
        atoms = self.__class__(Z=Z, Zval=Zval, pos=pos, cell = cell, basis = self.pos.basis)
        return atoms


z2lab = [
    "NA",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
