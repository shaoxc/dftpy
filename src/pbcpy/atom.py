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
            PP_file=None, basis='Cartesian', PME = True, BsplineOrder = 10):
            # PP_file=None, basis='Cartesian', PME = False, BsplineOrder = 10):
        '''
        Atom class handles atomic position, atom type and local pseudo potentials.
        '''

        if Zval is None :
            self.Zval = {}
        else :
            self.Zval = Zval
        # self.pos = Coord(pos, cell, basis='Cartesian')
        self.pos = Coord(pos, cell, basis=basis).to_cart()

        # private vars
        self._gp = {}          # 1D PP grid g-space
        self._vp = {}          # PP on 1D PP grid
        self._alpha_mu = {}    # G=0 of PP
        self._vloc_interp = {} # Interpolates recpot PP
        self._vlines = {}
        self._v = None         # PP for atom on 3D PW grid
        self._vreal = None     # PP for atom on 3D real space
        #
        self.nat = len(pos)
        self.usePME = PME
        self.BsplineOrder = BsplineOrder
        self.PP_file = PP_file
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

    def restart(self):
        '''
        clean all saved data
        '''
        self._gp = {}          # 1D PP grid g-space
        self._vp = {}          # PP on 1D PP grid
        self._alpha_mu = {}    # G=0 of PP
        self._vloc_interp = {} # Interpolates recpot PP
        self._vlines = {}
        self._v = None         # PP for atom on 3D PW grid
        self._vreal = None     # PP for atom on 3D real space

    def init_PP(self,PP_file = None):   
        if PP_file is None:
            PP_file = self.PP_file
        for key in PP_file :
            if not os.path.isfile(PP_file[key]):
                print("PP file not found")
                return Exception
            else :
                gp, vp = self.set_PP(PP_file[key])
                self._gp[key] = gp
                self._vp[key] = vp
                self._alpha_mu[key] = vp[0]
                vloc_interp = self.interpolate_PP(gp, vp)
                self._vloc_interp[key] = vloc_interp
        return 

    def set_PP(self,PP_file):
        '''Reads CASTEP-like recpot PP file
        Returns tuple (g, v)'''
        # HARTREE2EV = 27.2113845
        # BOHR2ANG   = 0.529177211
        HARTREE2EV = ENERGY_CONV['Hartree']['eV']
        BOHR2ANG   = LEN_CONV['Bohr']['Angstrom']
        with open(PP_file,'r') as outfil:
            lines = outfil.readlines()

        for i in range(0,len(lines)):
            line = lines[i]
            if 'END COMMENT' in line:
                ibegin = i+3
            elif line.strip() == '1000' :
            # if '  1000' in line:
                iend = i
        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        if '1000' in lines[iend]:
            print('Recpot pseudopotential '+PP_file+' loaded')
        else:
            return Exception
        gmax = np.float(lines[ibegin-1].strip())*BOHR2ANG
        v = np.array(line.split()).astype(np.float)/HARTREE2EV/BOHR2ANG**3
        g = np.linspace(0,gmax,num=len(v))
        return g, v


    def interpolate_PP(self,g_PP,v_PP,order=3):
        '''Interpolates recpot PP
        Returns interpolation function
        Linear interpolation is the default.
        However, it can use 2nd and 3rd order interpolation
        by specifying order=n, n=1-3 in argument list.'''
        # return interp1d(g_PP,v_PP,kind=order)
        # return splrep(g_PP,v_PP,k=order)
        return splrep(g_PP,v_PP, k=order)


    def strf(self,reciprocal_grid, iatom):
        '''
        Returns the Structure Factor associated to i-th ion.
        '''
        a=np.exp(-1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos[iatom]))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])

    def istrf(self,reciprocal_grid, iatom):
        a=np.exp(1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos[iatom]))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])

    def local_PP(self,grid,rho,PP_file, calcType = 'Both'):
        '''
        Reads and interpolates the local pseudo potential.
        INPUT: grid, rho, and path to the PP file
        OUTPUT: Functional class containing 
            - local pp in real space as potential 
            - v*rho as energy density.
        '''
        if self._v is None:
            if self.usePME :
                self.Get_PP_Reciprocal_PME(grid,PP_file)
            else :
                self.Get_PP_Reciprocal(grid,PP_file)
        if self._vreal is None:
            self._vreal = DirectField(grid=grid,griddata_3d=self._v.ifft(force_real=True))
        pot = self._vreal
        if calcType == 'Energy' or calcType == 'Both' :
            ene = np.einsum('ijkl, ijkl->', self._vreal , rho) * rho.grid.dV
        else :
            ene = 0
        return Functional(name='eN',energy=ene, potential=pot)

    def Get_PP_Reciprocal(self,grid,PP_file = None):   
        TimeData.Begin('Vion')

        reciprocal_grid = grid.get_reciprocal()
        q = np.sqrt(reciprocal_grid.gg)

        # v = 1j * np.zeros_like(q)
        v = np.zeros_like(q, dtype = np.complex128)
        vloc = np.empty_like(q)
        if not self._vloc_interp :
            self.init_PP(PP_file)
        labels = self._vloc_interp.keys()
        for key in labels:
            vloc_interp = self._vloc_interp[key]
            vloc[:] = 0.0
            # vloc[q<np.max(gp)] = vloc_interp(q[q<np.max(gp)])
            vloc[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 0)
            self._vlines[key] = vloc
            for i in range(len(self.pos)):
                if self.labels[i] == key :
                    strf = self.strf(reciprocal_grid, i)
                    v += vloc * strf
        self._v = ReciprocalField(reciprocal_grid,griddata_3d=v)
        TimeData.End('Vion')
        return "PP successfully interpolated"

    def Get_PP_Reciprocal_PME(self,grid,PP_file):   
        TimeData.Begin('Vion_PME')
        self.Bspline = CBspline(ions = self, grid = grid, order = self.BsplineOrder)
        reciprocal_grid = grid.get_reciprocal()
        q = np.sqrt(reciprocal_grid.gg)

        # v = 1j * np.zeros_like(q)
        v = np.zeros_like(q, dtype = np.complex128)
        # Qarray = DirectField(grid=grid,griddata_3d=np.zeros_like(q), rank=1)
        QA = np.empty(grid.nr)
        if not self._vloc_interp :
            self.init_PP(PP_file)
        labels = self._vloc_interp.keys()
        for key in labels:
            vloc_interp = self._vloc_interp[key]
            vloc = np.zeros(np.shape(q))
            vloc[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 0)
            self._vlines[key] = vloc
            QA[:] = 0.0
            for i in range(len(self.pos)):
                if self.labels[i] == key :
                    # Qarray += self.Bspline.get_PME_Qarray(i)
                    QA = self.Bspline.get_PME_Qarray(i, QA)
            Qarray = DirectField(grid=grid,griddata_3d=QA, rank=1)
            v = v + vloc * Qarray.fft()
        v = v * self.Bspline.Barray
        v *= grid.nnr / grid.volume
        self._v = ReciprocalField(reciprocal_grid,griddata_3d=v)
        TimeData.End('Vion_PME')
        return "PP successfully interpolated"

    def Get_PP_Derivative_One(self, grid, key = None):
        reciprocal_grid = grid.get_reciprocal()
        q = np.sqrt(reciprocal_grid.gg)
        # gp = self._gp[key]
        # vp = self._vp[key]
        # vloc_interp = self.interpolate_PP(gp, vp)
        vloc_interp = self._vloc_interp[key]
        vloc_deriv = np.zeros(np.shape(q))
        vloc_deriv[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 1)
        return ReciprocalField(reciprocal_grid,griddata_3d=vloc_deriv)

    def Get_PP_Derivative(self, grid, labels = None):
        reciprocal_grid = grid.get_reciprocal()
        q = np.sqrt(reciprocal_grid.gg)
        # v = 1j * np.zeros_like(q)
        v = np.zeros_like(q, dtype = np.complex128)
        vloc_deriv = np.empty_like(q, dtype = np.complex128)
        if labels is None :
            labels = self._gp.keys()
        for key in labels :
            # gp = self._gp[key]
            # vp = self._vp[key]
            # vloc_interp = self.interpolate_PP(gp, vp)
            vloc_interp = self._vloc_interp[key]
            vloc_deriv[:] = 0.0
            # vloc_deriv[q<np.max(gp)] = splev(q[q<np.max(gp)], vloc_interp, der = 1)
            vloc_deriv[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 1)
            for i in range(len(self.pos)):
                if self.labels[i] == key :
                    strf = self.strf(reciprocal_grid, i)
                    v += vloc_deriv * np.conjugate(strf)
        return ReciprocalField(reciprocal_grid,griddata_3d=v)



    @property
    def v(self):
        if self._v is not None:
            return self._v
        else:
            return Exception("Must load PP first")

    @property
    def vlines(self):
        if self._vlines is not None:
            return self._vlines
        else:
            return Exception("Must load PP first")

    @property
    def alpha_mu(self):
        if self._alpha_mu is not None:
            return self._alpha_mu
        else:
            return Exception("Must define PP before requesting alpha_mu")

    def set_Zval(self, labels = None):
        if labels is None :
            labels = self._gp.keys()
        for key in labels :
            gp = self._gp[key]
            vp = self._vp[key]
            val = (vp[0]-vp[1]) * (gp[-1]/(gp.size - 1)) ** 2 / (4.0 * np.pi)
            self.Zval[key] = round(val)
         
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


