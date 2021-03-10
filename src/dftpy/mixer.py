import numpy as np
import scipy.special as sp
from scipy import ndimage
from scipy.optimize import minpack2
from scipy import optimize as sopt
import time

from abc import ABC, abstractmethod
import warnings


class AbstractMixer(ABC):
    """
    This is a template class for mixer
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class LinearMixer(AbstractMixer):
    def __init__(self, coef = None, orthogonal = True, predcond = 'kerker', predcoef = [0.8, 0.1]):
        self.coef = coef
        self.orthogonal= orthogonal
        self.predcond = predcond
        self.predcoef = predcoef

    def __call__(self, arr1, arr2, coef = [0.5]):
        if self.predcond is None :
            if self.orthogonal :
                a1 = np.cos(coef[0])
                a2 = np.sin(coef[0])
            elif len(coef) == 1 :
                a1 = 0.0
                a2 = coef[0]
            else :
                a1 = coef[0]
                a2 = coef[1]
            results = arr1 * a1 + arr2 * a2
        elif self.predcond == 'kerker' :
            A = self.predcoef[0]
            q0 = self.predcoef[1] ** 2
            arr1G = arr1.fft()
            arr2G = arr2.fft()
            gg = arr1.grid.get_reciprocal().gg
            results = arr1G + A * gg/(gg+q0)*arr2G
            results = results.ifft(force_real=True)
        return results
