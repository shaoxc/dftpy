# Class handling output of functional evaluations
from .grid import DirectGrid, ReciprocalGrid
from .field import DirectField, ReciprocalField

# general python imports
import numpy as np

class Functional(object):
    '''
    Object handling DFT functional output
    
    Attributes
    ----------
    name: string
        The (optional) name of the functional

    energydensity: DirectField
        The energy density 

    potential: DirectField
        The first functional derivative of the functional wrt 
        the electron density 
        
    kernel: ReciprocalField
        The value of the reciprocal space kernel. This will
        be populated only if the functional is nonlocal
    '''


    def __init__(self, name=None, energydensity=None, potential=None, kernel=None):
        
        if name is not None:
            self.name = name
        else:
            raise AttributeError('Functional name must be specified')
        
        if energydensity is not None:
            if isinstance(energydensity, DirectField):
                self.energydensity = energydensity
            else:
                print('Cazzarola!')
        if potential is not None:
            if isinstance(potential, DirectField):
                self.potential = potential
        if kernel is not None:
            if isinstance(kernel, (np.ndarray)):
                self.kernel = kernel


    def sum(self,other):
        energydensity = self.energydensity+other.energydensity
        potential = self.potential+other.potential
        name = self.name + other.name
        return Functional(name=name,energydensity=energydensity,potential=potential)
    
    def __add__(self,other):
        return self.sum(other)




