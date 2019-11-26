# Collection of local and semilocal functionals

import numpy as np
from dftpy.field import DirectField,ReciprocalField
from dftpy.functional_output import Functional
from dftpy.math_utils import TimeData, PowerInt
from dftpy.kedf.tf import TF

def GGA(rho, functional = 'LKT', calcType = 'Both', split = False, **kwargs):
    pass
