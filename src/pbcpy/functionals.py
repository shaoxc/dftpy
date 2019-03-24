# Class handling functional evaluations 
# functional class (output handler) in output

# local imports
from .grid import DirectGrid, ReciprocalGrid
from .field import DirectField, ReciprocalField
from .functional_output import Functional
from .semilocal_xc import PBE, LDA, XC, KEDF
from .local_functionals_utils import TF,vW, x_TF_y_vW
from .local_pseudopotential import NuclearElectron
from .hartree import HartreeFunctional

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
    def ComputeEnergyDensityPotential(self,rho,**kwargs):
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
     outXC.energydensity --> the edens
     outXC.potential     --> the pot
    '''


    def __call__(self,rho):
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
        return self.ComputeEnergyDensityPotential(rho)
    
    def __init__(self,type=None,name=None,is_nonlocal=None,optional_kwargs=None):
        #init the class
        
        if optional_kwargs is None:
            self.optional_kwargs = { }
        else:
            self.optional_kwargs = optional_kwargs
        
        self.FunctionalNameList = []
        self.FunctionalTypeList = []
        
        self.FunctionalTypeList = ['XC','KEDF','IONS','HARTREE']
        XCNameList = ['LDA','PBE','LIBXC_XC','CUSTOM_XC']
        KEDFNameList = ['TF','vW','x_TF_y_vW','LC94','revAPBEK','TFvW','LIBXC_KEDF','CUSTOM_KEDF']
        KEDFNLNameList = ['WT','MGP','MGP0','WGC2','WGC1','WGC0','LMGP','LMGP0','LWT']
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

        if is_nonlocal is None:
            if type not in ['HARTREE','IONS']:
                raise AttributeError('Must assign is_nonlocal to FunctionalClass')
            else:
                self.is_nonlocal=False
        else:
            self.is_nonlocal = is_nonlocal
            
        if not isinstance(self.optional_kwargs,dict):
            raise AttributeError('optional_kwargs must be dict')
            
        if not self.CheckFunctional():
            raise Exception ('Functional check failed') 
    
    def ComputeEnergyDensityPotential(self,rho):
        if self.type == 'KEDF':
            if self.name == 'TF':
                return TF(rho)
            if self.name == 'vW':
                Sigma = self.optional_kwargs.get('Sigma',0.025)
                return vW(rho=rho,Sigma=Sigma)
            if self.name == 'x_TF_y_vW':
                Sigma = self.optional_kwargs.get('Sigma',0.025)
                x = self.optional_kwargs.get('x',1.0)
                y = self.optional_kwargs.get('x',0.0)
                return x_TF_y_vW(rho=rho,x=x,y=y,Sigma=Sigma)
            if self.name == 'LC94':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                return KEDF(density=rho,polarization=polarization,k_str='gga_k_lc94')
            if self.name == 'LIBXC_KEDF':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                k_str = optional_kwargs.get('k_str','gga_k_lc94')
                return KEDF(density=rho,polarization=polarization,k_str=k_str)
            if self.is_nonlocal == True:
                raise Exception('Nonlocal KEDF to be implemented')
        if self.type == 'XC':
            if self.name == 'LDA':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                return LDA(density=rho,polarization=polarization)
            if self.name == 'PBE':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                return PBE(density=rho,polarization=polarization)
            if self.name == 'LIBXC_XC':
                polarization = self.optional_kwargs.get('polarization','unpolarized')
                x_str = self.optional_kwargs.get('x_str','gga_x_pbe')
                c_str = self.optional_kwargs.get('c_str','gga_c_pbe')
                return XC(density=rho,x_str=x_str,c_str=c_str,polarization=polarization)
        if self.type == 'HARTREE':
            return HartreeFunctional(density=rho)
        if self.type == 'IONS':
            PP_list = self.optional_kwargs.get('PP_list')
            ions = self.optional_kwargs.get('ions')
            return NuclearElectron(density=rho,ions=ions,PPs=PP_list)



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
     out = EnergyEvaluator.ComputeEnergyDensityPotential(rho)
     edens = out.energydensity ...

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
        func = self.ComputeEnergyDensityPotential(rho_)
        E=func.energydensity.integral()
        int_tem_ = phi_*phi_*func.potential
        other_term_ = - int_tem_.integral() / N_
        final_v_ = ( func.potential + other_term_ ) * 2.0 * phi_  * self.N/N_ * rho_.grid.dV
        return  E , final_v_.ravel()
    
    def ComputeEnergyDensityPotential(self,rho):
        return self.KineticEnergyFunctional(rho) + self.XCFunctional(rho) + self.IONS(rho) + self.HARTREE(rho)

 
    def Energy(self,rho,ions):
        from .ewald import ewald
        ewald_ = ewald(rho=rho,ions=ions)
        total_edens =  self.KineticEnergyFunctional.ComputeEnergyDensityPotential(rho) + self.XCFunctional.ComputeEnergyDensityPotential(rho)  + self.HARTREE.ComputeEnergyDensityPotential(rho)
        return ewald_.energy + total_edens.energydensity.integral()





