import numpy as np
import scipy.special as sp
from scipy import ndimage
from scipy.optimize import minpack2
from scipy import optimize as sopt
import time
from dftpy.constants import FFTLIB

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

    @abstractmethod
    def restart(self):
        pass


class LinearMixer(AbstractMixer):
    def __init__(self, coef = None, orthogonal = False):
        self.coef = coef
        self.orthogonal= orthogonal

    def __call__(self, arr1, arr2, coef = [0.5]):
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
        return results
