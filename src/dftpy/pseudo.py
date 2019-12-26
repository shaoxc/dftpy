import os
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from dftpy.base              import Coord
from dftpy.field             import ReciprocalField, DirectField
from dftpy.functional_output import Functional
from dftpy.constants         import LEN_CONV, ENERGY_CONV
from dftpy.ewald             import CBspline
from dftpy.math_utils        import TimeData
from dftpy.atom              import Atom
from abc import ABC, abstractmethod
from dftpy.grid import DirectGrid, ReciprocalGrid
import warnings


# NEVER TOUCH THIS CLASS
# NEVER TOUCH THIS CLASS
class AbstractLocalPseudo(ABC):
    '''
    This is a pseudo potential template class and should never be touched.
    '''

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):   
        pass
    @abstractmethod
    def local_PP(self):   
        pass
    @abstractmethod
    def restart(self):   
        pass
    @abstractmethod
    def force(self):   
        pass
    @abstractmethod
    def stress(self):   
        pass
    @property
    @abstractmethod
    def v(self):   
        pass
    @property
    @abstractmethod
    def vreal(self):   
        pass
    @property
    @abstractmethod
    def vlines(self):   
        pass

# Only touch this class if you know what you are doing
class LocalPseudo(AbstractLocalPseudo):
    '''
    LocalPseudo class handles local pseudo potentials.
    This is a template class and should never be touched.
    '''
    def __init__(self, grid=None, ions=None, PP_list=None, PME=True):
        #
        # private vars
        self._gp = {}          # 1D PP grid g-space
        self._vp = {}          # PP on 1D PP grid
        self._vloc_interp = {} # Interpolates recpot PP
        self._vlines = {}
        self._v = None         # PP for atom on 3D PW grid
        self._vreal = None     # PP for atom on 3D real space
         
        if PME is not True:
            warnings.warn("Using N^2 method for strf!")

        self.usePME = PME
        
        if PP_list is not None:
            self.PP_list = PP_list
        else:
            raise AttributeError("Must specify PP_list for Pseudopotentials")
         
        if grid is not None:
            self.grid = grid
        else:
            raise AttributeError("Must specify Grid for Pseudopotentials")
         
        if not isinstance(self.grid,(DirectGrid)):
            raise TypeError("Grid must be DirectGrid")
         
        if ions is not None:
            self.ions = ions
        else:
            raise AttributeError("Must specify ions for Pseudopotentials")
         
        if not isinstance(self.ions,(Atom)):
            raise TypeError("Ions must be an array of Atom classes")
         
        if len(self.PP_list) != len(set(self.ions.Z)):
            raise ValueError("Incorrect number of pseudopotential files")

        if not self._vloc_interp :
            readPP = ReadPseudo(grid,PP_list)
            self._vloc_interp = readPP.vloc_interp
            self._vlines = readPP.vlines
            self._gp = readPP.gp
            self._vp = readPP.vp
            #update Zval in ions
            readPP.get_Zval(self.ions)

    def __call__(self,density=None,calcType=None):   
        if self._vreal is None:
            self.local_PP()
        pot = self._vreal
        if calcType == 'Energy' or calcType == 'Both' :
            ene = np.einsum('ijkl, ijkl->', self._vreal , density) * self.grid.dV
        else :
            ene = 0
        return Functional(name='eN',energy=ene, potential=pot)


    def local_PP(self, BsplineOrder = 10):
        '''
        '''
        if BsplineOrder is not 10:
            warnings.warn("BsplineOrder not 10. Do you know what you are doing?")
        self.BsplineOrder = BsplineOrder

        if self._v is None:
            if self.usePME :
                self._PP_Reciprocal_PME()
            else :
                self._PP_Reciprocal()
        if self._vreal is None:
            self._vreal = self._v.ifft(force_real=True)


    def restart(self, grid=None, ions=None, full=False):
        '''
        Clean all private data and resets the ions and grid.
        This will prompt the computation of a new pseudo 
        without recomputing the local pp on the atoms.
        '''
        if full :
            self._gp = {}          # 1D PP grid g-space
            self._vp = {}          # PP on 1D PP grid
            self._vloc_interp = {} # Interpolates recpot PP
        self._vlines = {}
        self._v = None         # PP for atom on 3D PW grid
        self._vreal = None     # PP for atom on 3D real space
        if ions is not None:
            self.ions = ions
        if grid is not None:
            self.grid = grid


    def stress(self,rho,energy):
        if rho is None:
            raise AttributeError("Must specify rho")
        if not isinstance(rho,(DirectField)):
            raise TypeError("rho must be DirectField")
        if self.usePME:
            return self._StressPME(rho,energy)
        else:
            return self._Stress(rho,energy)


    def force(self,rho):
       if rho is None:
           raise AttributeError("Must specify rho")
       if not isinstance(rho,(DirectField)):
           raise TypeError("rho must be DirectField")
       if self.usePME:
           return self._ForcePME(rho)
       else:
           return self._Force(rho)

    @property
    def vreal(self):
        '''
        The vloc represented on the real space grid.
        '''
        if self._vreal is not None:
            return self._vreal
        else:
            return Exception("Must load PP first")

    @property
    def v(self):
        '''
        The vloc represented on the reciprocal space grid.
        '''
        if self._v is not None:
            return self._v
        else:
            return Exception("Must load PP first")

    @property
    def vlines(self):
        '''
        The vloc for each atom type represented on the reciprocal space grid.
        '''
        if self._vlines:
            return self._vlines
        else:
            return Exception("Must load PP first")

    def _PP_Reciprocal(self):   
        TimeData.Begin('Vion')
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype = np.complex128)
        for key in self._vloc_interp.keys():
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key :
                    strf = self.ions.strf(reciprocal_grid, i)
                    v += self._vlines[key] * strf
        self._v = ReciprocalField(reciprocal_grid,griddata_3d=v)
        TimeData.End('Vion')
        return "PP successfully interpolated"

    def _PP_Reciprocal_PME(self):   
        TimeData.Begin('Vion_PME')
        self.Bspline = CBspline(ions = self.ions, grid = self.grid, order = self.BsplineOrder)
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype = np.complex128)
        QA = np.empty(self.grid.nr)
        for key in self._vloc_interp.keys():
            QA[:] = 0.0
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key :
                    QA = self.Bspline.get_PME_Qarray(i, QA)
            Qarray = DirectField(grid=self.grid,griddata_3d=QA, rank=1)
            v = v + self._vlines[key] * Qarray.fft()
        v = v * self.Bspline.Barray * self.grid.nnr / self.grid.volume
        self._v = v
        TimeData.End('Vion_PME')
        return "PP successfully interpolated"

    def _PP_Derivative_One(self, key = None):
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        vloc_interp = self._vloc_interp[key]
        vloc_deriv = np.zeros(np.shape(q))
        vloc_deriv[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 1)
        return ReciprocalField(reciprocal_grid,griddata_3d=vloc_deriv)

    def _PP_Derivative(self, labels = None):
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype = np.complex128)
        vloc_deriv = np.empty_like(q, dtype = np.complex128)
        if labels is None :
            labels = self._gp.keys()
        for key in labels :
            vloc_interp = self._vloc_interp[key]
            vloc_deriv[:] = 0.0
            vloc_deriv[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 1)
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key :
                    strf = self.ions.strf(reciprocal_grid, i)
                    v += vloc_deriv * np.conjugate(strf)
        return v

    def _Stress(self,rho,energy=None):
        if energy is None :
            energy = self(density=rho, calcType='Energy').energy
        reciprocal_grid=self.grid.get_reciprocal()
        g= reciprocal_grid.g
        #gg= reciprocal_grid.gg
        mask = reciprocal_grid.mask
        mask2 = mask[..., np.newaxis]
        q = reciprocal_grid.q
        q[0, 0, 0, 0] = 1.0
        rhoG = rho.fft()
        stress = np.zeros((3, 3))
        v_deriv=self._PP_Derivative()
        rhoGV_q = rhoG * v_deriv / q
        for i in range(3):
            for j in range(i, 3):
                # den = (g[..., i]*g[..., j])[..., np.newaxis] * rhoGV_q
                # stress[i, j] = (np.einsum('ijkl->', den)).real / rho.grid.volume
                den = (g[..., i][mask]*g[..., j][mask]) * rhoGV_q[mask2]
                stress[i, j] = stress[j, i] = -(np.einsum('i->', den)).real / self.grid.volume*2.0
                if i == j :
                    stress[i, j] -= energy
        stress /= self.grid.volume
        return stress
    
    def _Force(self,rho):
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        Forces= np.zeros((self.ions.nat, 3))
        mask = reciprocal_grid.mask
        mask2 = mask[..., np.newaxis]
        for i in range(self.ions.nat):
            strf = self.ions.istrf(reciprocal_grid, i)
            den = self.vlines[self.ions.labels[i]][mask2]* (rhoG[mask2] * strf[mask2]).imag
            for j in range(3):
                Forces[i, j] = np.einsum('i, i->', g[..., j][mask], den)
        Forces *= 2.0/self.grid.volume
        return Forces
    
    def _ForcePME(self, rho):
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        Bspline = self.Bspline
        Barray = Bspline.Barray
        Barray = np.conjugate(Barray)
        denG = rhoG * Barray
        nr = self.grid.nr
        #cell_inv = np.linalg.inv(self.ions.pos[0].cell.lattice)
        cell_inv = reciprocal_grid.lattice/2/np.pi
        Forces= np.zeros((self.ions.nat, 3))
        ixyzA = np.mgrid[:self.BsplineOrder, :self.BsplineOrder, :self.BsplineOrder].reshape((3, -1))
        Q_derivativeA = np.zeros((3, self.BsplineOrder * self.BsplineOrder * self.BsplineOrder))
        for key in self.ions.Zval.keys():
            denGV = denG * self.vlines[key]
            denGV[0, 0, 0, 0] = 0.0+0.0j
            rhoPB = denGV.ifft(force_real = True)[..., 0]
            for i in range(self.ions.nat):
                if self.ions.labels[i] == key :
                    Up = np.array(self.ions.pos[i].to_crys()) * nr
                    Mn = []
                    Mn_2 = []
                    for j in range(3):
                        Mn.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j])) )
                        Mn_2.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order = self.BsplineOrder - 1) )
                    Q_derivativeA[0] = nr[0] * np.einsum('i, j, k -> ijk', Mn_2[0][1:]-Mn_2[0][:-1], Mn[1][1:], Mn[2][1:]).reshape(-1)
                    Q_derivativeA[1] = nr[1] * np.einsum('i, j, k -> ijk', Mn[0][1:], Mn_2[1][1:]-Mn_2[1][:-1], Mn[2][1:]).reshape(-1)
                    Q_derivativeA[2] = nr[2] * np.einsum('i, j, k -> ijk', Mn[0][1:], Mn[1][1:], Mn_2[2][1:]-Mn_2[2][:-1]).reshape(-1)
                    l123A = np.mod(1+np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
                    Forces[i] = -np.sum(np.matmul(Q_derivativeA.T, cell_inv) * rhoPB[l123A[0], l123A[1], l123A[2]][:, np.newaxis], axis=0)
        return Forces


    def _StressPME(self, rho,energy=None):
        if energy is None :
            energy = self(density=rho, calcType='Energy').energy
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        q = reciprocal_grid.q
        q[0, 0, 0, 0] = 1.0
        mask = reciprocal_grid.mask
        mask2 = mask[..., np.newaxis]
        Bspline = self.Bspline
        Barray = Bspline.Barray
        rhoGB = np.conjugate(rhoG) * Barray
        nr = self.grid.nr
        stress = np.zeros((3, 3))
        QA = np.empty(nr)
        for key in self.ions.Zval.keys():
            rhoGBV = rhoGB * self._PP_Derivative_One(key = key)
            QA[:] = 0.0
            for i in range(self.ions.nat):
                if self.ions.labels[i] == key :
                    QA = self.Bspline.get_PME_Qarray(i, QA)
            Qarray = DirectField(grid=self.grid,griddata_3d=QA, rank=1)
            rhoGBV = rhoGBV * Qarray.fft()
            for i in range(3):
                for j in range(i, 3):
                    den = (g[..., i][mask]*g[..., j][mask]) * rhoGBV[mask2]/ q[mask2]
                    stress[i, j] -= (np.einsum('i->', den)).real / self.grid.volume**2
        stress *= 2.0 * self.grid.nnr
        for i in range(3):
            for j in range(i, 3):
                stress[j, i] = stress[i, j]
            stress[i, i] -= energy
        stress /= self.grid.volume
        return stress

class ReadPseudo(object):
    '''
    Support class for LocalPseudo.
    '''
    def __init__(self, grid=None, PP_list=None):
        self._gp = {}          # 1D PP grid g-space
        self._vp = {}          # PP on 1D PP grid
        self._vloc_interp = {} # Interpolates recpot PP
        self._vlines = {}
        self._upf = {}

        self.PP_list = PP_list
        self.grid = grid
        key = list(self.PP_list.keys())[0]
        if PP_list[key][-6:].lower() == 'recpot':
            self.PP_type = 'recpot'
        elif PP_list[key][-3:].lower() == 'upf':
            self.PP_type = 'upf'
        else:
            raise Exception("Pseudopotential not supported")
        if self.PP_type is 'recpot':
            self._init_PP_recpot()
        elif self.PP_type is 'upf':
            self._init_PP_upf()


    def _init_PP_recpot(self):
        '''
        This is a private method used only in this specific class. 
        '''
        def set_PP(Single_PP_file):
            '''Reads CASTEP-like recpot PP file
            Returns tuple (g, v)'''
            # HARTREE2EV = 27.2113845
            # BOHR2ANG   = 0.529177211
            HARTREE2EV = ENERGY_CONV['Hartree']['eV']
            BOHR2ANG   = LEN_CONV['Bohr']['Angstrom']
            with open(Single_PP_file,'r') as outfil:
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
                print('Recpot pseudopotential '+Single_PP_file+' loaded')
            else:
                return Exception
            gmax = np.float(lines[ibegin-1].strip())*BOHR2ANG
            v = np.array(line.split()).astype(np.float)/HARTREE2EV/BOHR2ANG**3
            g = np.linspace(0,gmax,num=len(v))
            return g, v
        for key in self.PP_list :
            print('setting key: '+key)
            if not os.path.isfile(self.PP_list[key]):
                raise Exception("PP file for atom type "+str(key)+" not found")
            else :
                gp, vp = set_PP(self.PP_list[key])
                self._gp[key] = gp
                self._vp[key] = vp
                vloc_interp = splrep(gp, vp)
                self._vloc_interp[key] = vloc_interp
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        vloc = np.empty_like(q)
        for key in self._vloc_interp.keys():
            vloc_interp = self._vloc_interp[key]
            vloc[:] = 0.0
            vloc[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 0)
            self._vlines[key] = vloc


    def _init_PP_upf(self):
        '''
        This is a private method used only in this specific class. 
        '''
        def set_PP(Single_PP_file,MaxPoints=1000,Gmax=60):
            '''Reads QE UPF type PP'''
            import importlib
            upf2json = importlib.util.find_spec("upf_to_json")
            found = upf2json is not None
            if found:
                from upf_to_json import upf_to_json
            else: 
                raise ModuleNotFoundError("Must pip install upf_to_json") 
            Ry2Ha = ENERGY_CONV['Rydberg']['Hartree']
            with open(Single_PP_file,'r') as outfil:
                upf = upf_to_json(upf_str=outfil.read(),fname=Single_PP_file)
            r = np.array(upf['pseudo_potential']['radial_grid'],dtype=np.float64)
            v = np.array(upf['pseudo_potential']['local_potential'],dtype=np.float64)/Ry2Ha            
            return r, v, upf
        for key in self.PP_list :
            print('setting key: '+key)
            if not os.path.isfile(self.PP_list[key]):
                raise Exception("PP file for atom type "+str(key)+" not found")
            else :
                r , vr, self._upf[key] = set_PP(self.PP_list[key])
        
                gp = np.linspace(start=0,stop=Gmax,num=MaxPoints)
                vp = np.zeros_like(gp)
                vp[0] = 0.0
                for k in np.arange(start=1,stop=len(gp)):
                    vp[k] = (4.0*np.pi / gp[k]) * np.sum(r * vr* np.sin( gp[k] * r ))
                self._gp[key] = gp
                self._vp[key] = vp
                vloc_interp = splrep(gp, vp)
                self._vloc_interp[key] = vloc_interp
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        vloc = np.empty_like(q)
        for key in self._vloc_interp.keys():
            vloc_interp = self._vloc_interp[key]
            vloc[:] = 0.0
            vloc[q<np.max(self._gp[key])] = splev(q[q<np.max(self._gp[key])], vloc_interp, der = 0)
            self._vlines[key] = vloc.copy()

    def get_Zval(self,ions):
        if self.PP_type is 'upf':
            for key in self._gp.keys():
                ions.Zval[key] = self._upf[key]['pseudo_potential']['header']['z_valence'] 
        elif self.PP_type is 'recpot':
            for key in self._gp.keys():
                gp = self._gp[key]
                vp = self._vp[key]
                val = (vp[0]-vp[1]) * (gp[-1]/(gp.size - 1)) ** 2 / (4.0 * np.pi)
                ions.Zval[key] = round(val)


    @property
    def vloc_interp(self):
        if not self._vloc_interp:
            raise Exception("Must init ReadPseudo")
        return self._vloc_interp

    @property
    def vlines(self):
        if not self._vlines:
            raise Exception("Must init ReadPseudo")
        return self._vlines
        
    @property
    def gp(self):
        if not self._gp:
            raise Exception("Must init ReadPseudo")
        return self._gp

    @property
    def vp(self):
        if not self._vp:
            raise Exception("Must init ReadPseudo")
        return self._vp




