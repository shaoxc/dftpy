# Class handling functional evaluations 
# functional class (output handler) in output

# local imports
from dftpy.field import DirectField
from dftpy.functional_output import Functional
from dftpy.semilocal_xc import PBE, LDA, XC, KEDF
from dftpy.local_pseudopotential import NuclearElectron
from dftpy.hartree import HartreeFunctional
from dftpy.kedf import TF,vW, x_TF_y_vW, WT, LWT, FP, SM, MGP

# general python imports
from abc import ABC, abstractmethod
import numpy as np


class AbstractFunctional(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__ (self,rho,**kwargs):
        # call the XC and such... depending on kwargs
        # Interface for scipy.optimize
        pass
    
    @abstractmethod
    def ComputeEnergyPotential(self,rho,**kwargs):
        # returns edens and pot
        pass

    def GetName(self):
        return self.name

    def GetType(self):
        return self.type
    
    def AssignName(self,name):
        self.name = name

    def AssignType(self,type):
        self.type = type

    def CheckFunctional(self):
        if self.type not in self.FunctionalTypeList:
            print(self.type,' is not a valid Functional type')
            print('Valid Functional types are:')
            print(self.FunctionalTypeList)
            return False
        if self.name not in self.FunctionalNameList:
            print(self.name, ' is not a valid Functional name')
            print('Valid Functional names are:')
            print(self.FunctionalNameList)
            return False
        return True


class FunctionalClass(AbstractFunctional):
    '''
    Object handling evaluation of a DFT functional
    
    Attributes
    ----------
    name: string
        The name of the functional

    type: string
        The functional type (XC, KEDF, HARTREE, IONS) 

    is_nonlocal: logical
        Is the functional a nonlocal functional? 
        
    optional_kwargs: dict
        set of kwargs for the different functional types/names

 
    Example
    -------
     XC = FunctionalClass(type='XC',name='LDA')
     outXC = XC(rho)
     outXC.energy --> the energy
     outXC.potential     --> the pot
    '''


    def __call__(self,rho, calcType = 'Both'):
        '''
        Functional class is callable

        Attributes 
        ----------  
          rho: DirectField
             The input density

        Returns
        -------
          Functional: functional output handler
             The output is a Functional class
        '''
        return self.ComputeEnergyPotential(rho, calcType)
    
    def __init__(self,type=None,name=None,is_nonlocal=None,optional_kwargs=None,  **kwargs):
        #init the class
        
        if optional_kwargs is None:
            self.optional_kwargs = { }
        else:
            self.optional_kwargs = optional_kwargs
        self.optional_kwargs.update(kwargs)
        
        self.FunctionalNameList = []
        self.FunctionalTypeList = []
        
        self.FunctionalTypeList = ['XC','KEDF','IONS','HARTREE']
        XCNameList = ['LDA','PBE','LIBXC_XC','CUSTOM_XC']
        KEDFNameList = ['TF','vW','x_TF_y_vW','LC94','revAPBEK','TFvW','LIBXC_KEDF','CUSTOM_KEDF']
        KEDFNLNameList = ['WT','MGP','MGP0','WGC2','WGC1','WGC0','LMGP','LMGP0','LWT', 'FP', 'SM', \
                'MGPA', 'MGPG', 'LMGP0', 'LMGPA', 'LMGPG']
        IONSNameList = ['IONS']
        HNameList = ['HARTREE']
        
        self.FunctionalNameList = XCNameList + KEDFNameList + KEDFNLNameList + IONSNameList +HNameList
        
        if type is None:
            raise AttributeError('Must assign type to FunctionalClass')
        else:
            self.type = type

        if name is None:
            if type not in ['HARTREE','IONS']:
                raise AttributeError('Must assign name to FunctionalClass')
            else:
                self.name=self.type
        else:
            self.name = name

        # if is_nonlocal is None:
            # if type not in ['HARTREE','IONS']:
                # raise AttributeError('Must assign is_nonlocal to FunctionalClass')
            # else:
                # self.is_nonlocal=False
        # else:
            # self.is_nonlocal = is_nonlocal
            
        if not isinstance(self.optional_kwargs,dict):
            raise AttributeError('optional_kwargs must be dict')
            
        if not self.CheckFunctional():
            raise Exception ('Functional check failed') 
    
    def ComputeEnergyPotential(self,rho, calcType = 'Both',  **kwargs):
        self.optional_kwargs.update(kwargs)
        if self.type == 'KEDF':
            if self.name == 'TF':
                return TF(rho, calcType=calcType, **self.optional_kwargs)
            elif self.name == 'vW':
                return vW(rho=rho, calcType=calcType, **self.optional_kwargs)
            elif self.name == 'x_TF_y_vW':
                return x_TF_y_vW(rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'LC94':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                return KEDF(rho,polarization=polarization,k_str='gga_k_lc94', calcType=calcType)
            elif self.name == 'LIBXC_KEDF':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                k_str = optional_kwargs.get('k_str','gga_k_lc94')
                return KEDF(rho,polarization=polarization,k_str=k_str, calcType=calcType)
            elif self.name == 'WT' :
                return WT(rho=rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'SM' :
                return SM(rho=rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'FP' :
                return FP(rho=rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'MGP' :
                return MGP(rho=rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'MGPA' :
                self.optional_kwargs['symmetrization'] = 'Arithmetic'
                return MGP(rho=rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'MGPG' :
                self.optional_kwargs['symmetrization'] = 'Geometric'
                return MGP(rho=rho,calcType=calcType, **self.optional_kwargs)
            elif self.name == 'LWT' :
                self.optional_kwargs['kerneltype'] = 'WT'
                return LWT(rho=rho, calcType=calcType, **self.optional_kwargs)
            elif self.name == 'LMGP' :
                self.optional_kwargs['kerneltype'] = 'MGP'
                return LWT(rho=rho, calcType=calcType, **self.optional_kwargs)
            elif self.name == 'LMGPA' :
                self.optional_kwargs['kerneltype'] = 'MGPA'
                return LWT(rho=rho, calcType=calcType, **self.optional_kwargs)
            elif self.name == 'LMGPG' :
                self.optional_kwargs['kerneltype'] = 'MGPG'
                return LWT(rho=rho, calcType=calcType, **self.optional_kwargs)
            else :
                raise Exception(self.name + ' KEDF to be implemented')
            # if self.is_nonlocal == True:
                # raise Exception('Nonlocal KEDF to be implemented')
        if self.type == 'XC':
            if self.name == 'LDA':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                return LDA(rho,polarization=polarization, calcType=calcType, **self.optional_kwargs)
            if self.name == 'PBE':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                return PBE(density=rho,polarization=polarization, calcType=calcType, **self.optional_kwargs)
            if self.name == 'LIBXC_XC':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                x_str = self.optional_kwargs.get('x_str','gga_x_pbe')
                c_str = self.optional_kwargs.get('c_str','gga_c_pbe')
                return XC(density=rho,x_str=x_str,c_str=c_str,polarization=polarization, calcType=calcType, **self.optional_kwargs)
        if self.type == 'HARTREE':
            return HartreeFunctional(density=rho, calcType=calcType)
        if self.type == 'IONS':
            PP_list = self.optional_kwargs.get('PP_list')
            ions = self.optional_kwargs.get('ions')
            return NuclearElectron(density=rho,ions=ions,PPs=PP_list, calcType=calcType)

class TotalEnergyAndPotential(object):
    '''
     Object handling energy evaluation for the 
     purposes of optimizing the electron density
     
     Attributes
     ----------

     KineticEnergyFunctional, XCFunctional, IONS, HARTREE: FunctionalClass
         Instances of functional class needed for the computation
         of the chemical potential, total potential and total energy.

     Example
     -------

     XC = FunctionalClass(type='XC',name='LDA')
     KE = FunctionalClass(type='KEDF',name='TF')
     HARTREE = FunctionalClass(type='HARTREE')
     IONS = FunctionalClass(type='IONS', kwargs)

     EnergyEvaluator = TotalEnergyAndPotential(KEDF,XC,IONS,HARTREE,rho_guess)

     [the energy:]
     E = EnergyEvaluator.Energy(rho,ions)
     
     [total energy and potential:]
     out = EnergyEvaluator.ComputeEnergyPotential(rho)

     [time for optimization of density:]
     in_for_scipy_minimize = EnergyEvaluator(phi)
    '''
    
    def __init__(self,KineticEnergyFunctional=None, XCFunctional=None, IONS=None, HARTREE=None, rho=None):
        

        if KineticEnergyFunctional is None:
            raise AttributeError('Must define KineticEnergyFunctional')
        elif not isinstance(KineticEnergyFunctional, FunctionalClass):
            raise AttributeError('KineticEnergyFunctional must be FunctionalClass')
        else:
            self.KineticEnergyFunctional = KineticEnergyFunctional
                                 
        if XCFunctional is None:
            raise AttributeError('Must define XCFunctional')
        elif not isinstance(XCFunctional, FunctionalClass):
            raise AttributeError('XCFunctional must be FunctionalClass')
        else:
            self.XCFunctional = XCFunctional
                                 
        if IONS is None:
            raise AttributeError('Must define IONS')
        elif not isinstance(IONS, FunctionalClass):
            raise AttributeError('IONS must be FunctionalClass')
        else:
            self.IONS = IONS
                                 
        if HARTREE is None:
            print('WARNING: using FFT Hartree')
            self.HARTREE = HARTREE
        else:
            self.HARTREE = HARTREE
                                 
        if rho is None:
            raise AttributeError('Must define rho')
        elif not isinstance(rho, DirectField):
            raise AttributeError('rho must be DirectField')
        else:
            self.rho = rho
            self.N = self.rho.integral()
            
    def __call__ (self,phi):
        # call the XC and such... depending on kwargs
        rho_shape = np.shape(self.rho)
        if not isinstance(phi, DirectField):
            phi_ = DirectField(self.rho.grid,griddata_3d=np.reshape(phi,rho_shape),rank=1)
        else:
            phi_ = phi
        rho_ = phi_*phi_
        N_=rho_.integral()
        rho_ *= self.N/N_
        func = self.ComputeEnergyPotential(rho_)
        E=func.energy
        int_tem_ = phi_*phi_*func.potential
        other_term_ = - int_tem_.integral() / N_
        final_v_ = ( func.potential + other_term_ ) * 2.0 * phi_  * self.N/N_ * rho_.grid.dV
        return  E , final_v_.ravel()
    
    def ComputeEnergyPotential(self,rho, calcType = 'Both'):
        Obj = self.KineticEnergyFunctional(rho,calcType)\
                + self.XCFunctional(rho,calcType) + \
                self.IONS(rho,calcType) + self.HARTREE(rho,calcType)
        return Obj

 
    def Energy(self,rho,ions, usePME = False, calcType = 'Energy'):
        from .ewald import ewald
        ewald_ = ewald(rho=rho,ions=ions, PME = usePME)
        total_e = self.ComputeEnergyPotential(rho, calcType = 'Energy')
        # total_e= self.KineticEnergyFunctional.ComputeEnergyPotential(rho,calcType) + \
                # self.XCFunctional.ComputeEnergyPotential(rho,calcType) + \
                # self.HARTREE.ComputeEnergyPotential(rho,calcType) + \
                # self.IONS.ComputeEnergyPotential(rho,calcType)
        return ewald_.energy + total_e.energy
