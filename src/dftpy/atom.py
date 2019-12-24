import os
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from dftpy.base import Coord
from dftpy.field import ReciprocalField, DirectField
from dftpy.functional_output import Functional
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.ewald import CBspline
from dftpy.math_utils import TimeData

class Atom(object):

    def __init__(self, Z=None, Zval=None, label=None, pos=None, cell=None, \
            basis='Cartesian'):
        '''
        Atom class handles atomic position, atom type and local pseudo potentials.
        '''

        if Zval is None :
            self.Zval = {}
        else :
            self.Zval = Zval
        # self.pos = Coord(pos, cell, basis='Cartesian')
        self.pos = Coord(pos, cell, basis=basis).to_cart()
        self.nat = len(pos)
        self.labels = label
        self.Z = Z

        # check label 
        if self.labels :
            for i in range(len(self.labels)):
                if self.labels[i].isdigit():
                    self.labels[i] = z2lab[self.labels[i]]

        if self.Z is None:
            self.Z = []
            for item in self.labels :
                self.Z.append(z2lab.index(item))

        if self.labels is None:
            self.labels = []
            for item in self.Z :
                self.labels.append(z2lab[item])

    def set_Zval(self, labels = None):
        if self.Zval is None:
            raise Exception("Must initialize Pseudo Potential with ReadPseudo")
         
    def strf(self,reciprocal_grid, iatom):
        '''
        Returns the Structure Factor associated to i-th ion.
        '''
        a=np.exp(-1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos[iatom]))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])

    def istrf(self,reciprocal_grid, iatom):
        a=np.exp(1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos[iatom]))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])

z2lab = ['NA', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
         'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
         'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
         'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
         'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
         'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
         'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
         'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
         'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
         'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
         'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']


