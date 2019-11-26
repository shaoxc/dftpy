import numpy as np
import copy
from dftpy.constants import ENERGY_CONV, LEN_CONV
def DefaultOption(dicts = None):
    JOB = {
          'task' : 'Optdensity', 
          'calctype' : 'Energy'
          }

    PATH = {
            'pppath' : '', 
            'cellpath' : '' , 
            }

    MATH = {
            'linearii' : True, 
            'linearie' : True,
            'twostep' : False, 
            'reuse' : True,
            }

    PP = {}

    CELL = {
            'cellfile' : 'POSCAR', 
            'elename' : 'Al', 
            'zval' : None, 
            'format' : None, 
            }

    GRID = {
            'ecut': None, 
            'spacing': 0.25,
            'gfull':   False,
            'nr':      None,
            }

    DENSITY = {
            'densityini' : 'HEG', 
            'densityfile' : None,
            'densityoutput' : None, 
            }

    EXC = {
            'xc' : 'LDA', 
            'x_str' : 'lda_x',
            'c_str' : 'lda_c_pz', 
            }

    KEDF = {
            'kedf'           : 'WT',
            'x'              : 1.0,
            'y'              : 1.0,
            'alpha'          : 5.0/6.0,
            'beta'           : 5.0/6.0,
            'sigma'          : 0.025,
            'nsp'            : None, # The number of spline
            'interp'         : 'hermite',
            'kerneltype'     : 'WT',
            'symmetrization' : None,
            'lumpfactor'     : None, # factor for MGP
            'neta'           : 50000,
            'etamax'         : 50.0,
            'order'          : 3,
            'ratio'          : 1.2,
            'maxpoints'      : 1000,
            'delta'          : None, # The gap of spline
            'fd'             : 0, 
            }

    OUTPUT = {
            'time' : True, 
            'stress' : True,
            }

    OPT = {
            'method'  : 'CG-HS',
            'algorithm' : 'EMM', # Residual minimization method or Energy minimization method
            'vector' : 'Orthogonalization', # or Scaling
            'c1'      : 1e-4,
            'c2'      : 2e-1,
            'maxls'   : 10,
            'econv'   : 1e-6, # Energy Convergence (a.u./atom)
            'maxfun'  : 50,  # For TN method, it's the max steps for searching direction
            'maxiter' : 100,# The max steps for optimization
            'xtol'    : 1e-12, 
            'h0'     : 1.0,  # for LBFGS
            }

    conf = {
            'JOB' : JOB, 
            'PATH' : PATH, 
            'MATH' : MATH, 
            'PP' : PP, 
            'KEDF' :KEDF , 
            'CELL' : CELL, 
            'GRID' : GRID, 
            'EXC' : EXC, 
            'KEDF' : KEDF, 
            'OPT' : OPT, 
            'DENSITY' : DENSITY, 
            'OUTPUT' : OUTPUT
            }
    for section in conf :
        for key in conf[section] :
            conf[section][key] = str(conf[section][key])
    return conf

def OptionFormat(config):
    conf = copy.deepcopy(config)
    for section in conf :
        for key in conf[section] :
            if conf[section][key] == 'None' :
                conf[section][key] = None

    def bools(strings):
        s = strings.lower()[0]
        if s == 'n' or s == 'f' or s == '0' :
            return False
        else :
            return True

    conf['JOB']['task'] = conf['JOB']['task'].capitalize()
    conf['JOB']['calctype'] = [s.capitalize() for s in conf['JOB']['calctype'].split()]
    conf['MATH']['linearii'] = bools(conf['MATH']['linearii'])
    conf['MATH']['linearie'] = bools(conf['MATH']['linearie'])
    conf['MATH']['twostep'] = bools(conf['MATH']['twostep'])
    conf['MATH']['reuse'] = bools(conf['MATH']['reuse'])
    conf['KEDF']['x'] = float(eval(conf['KEDF']['x']))
    conf['KEDF']['y'] = float(eval(conf['KEDF']['y']))
    conf['KEDF']['alpha'] = float(eval(conf['KEDF']['alpha']))
    conf['KEDF']['beta'] = float(eval(conf['KEDF']['beta']))
    conf['KEDF']['Sigma'] = float(eval(conf['KEDF']['sigma']))
    conf['KEDF']['etamax'] = float(eval(conf['KEDF']['etamax']))
    conf['KEDF']['neta'] = int(conf['KEDF']['neta'])
    conf['KEDF']['order'] = int(conf['KEDF']['order'])
    conf['KEDF']['maxpoints'] = int(conf['KEDF']['maxpoints'])
    conf['KEDF']['ratio'] = float(eval(conf['KEDF']['ratio']))
    conf['KEDF']['fd'] = int(conf['KEDF']['fd'])
    if conf['KEDF']['nsp'] :
        conf['KEDF']['nsp'] = int(conf['KEDF']['nsp']) 
    if conf['KEDF']['delta'] :
        conf['KEDF']['delta'] = float(eval(conf['KEDF']['delta']))
    if conf['KEDF']['lumpfactor'] :
        l = conf['KEDF']['lumpfactor'].split()
        if len(l) > 1 :
            lump = [float(eval(item)) for item in l]
        else :
            lump = float(eval(l[0]))
        conf['KEDF']['lumpfactor'] = lump
        # conf['KEDF']['lumpfactor'] = float(eval(conf['KEDF']['lumpfactor']))

    conf['GRID']['spacing'] = float(eval(conf['GRID']['spacing']))
    if conf['GRID']['nr'] :
        conf['GRID']['nr'] = list(map(int,conf['GRID']['nr'].split()))

    if conf['CELL']['elename'] :
        conf['CELL']['elename'] = [s.capitalize() for s in conf['CELL']['elename'].split()]
    if conf['CELL']['zval'] :
        conf['CELL']['zval'] = list(map(float,conf['CELL']['zval'].split()))

    conf['OPT']['c1'] = float(conf['OPT']['c1'])
    conf['OPT']['c2'] = float(conf['OPT']['c2'])
    conf['OPT']['econv'] = float(eval(conf['OPT']['econv']))
    conf['OPT']['maxls'] = int(conf['OPT']['maxls'])
    conf['OPT']['maxfun'] = int(conf['OPT']['maxfun'])
    conf['OPT']['maxiter'] = int(conf['OPT']['maxiter'])
    conf['OPT']['xtol'] = float(eval(conf['OPT']['xtol']))
    conf['OPT']['h0'] = float(eval(conf['OPT']['h0']))

    conf['OUTPUT']['time'] = bools(conf['OUTPUT']['time'])
    conf['OUTPUT']['stress'] = bools(conf['OUTPUT']['stress'])
    ############################## Conversion of units  ##############################
    '''
    Ecut = pi^2/(2 * h^2)
    Ref : Briggs, E. L., D. J. Sullivan, and J. Bernholc. "Real-space multigrid-based approach to large-scale electronic structure calculations." Physical Review B 54.20 (1996): 14362.
    '''
    if conf['GRID']['ecut'] :
        conf['GRID']['ecut'] = float(eval(conf['GRID']['ecut'])) * ENERGY_CONV['eV']['Hartree']
        conf['GRID']['spacing'] = np.sqrt(np.pi ** 2/ conf['GRID']['ecut'] * 0.5)
    else :
        conf['GRID']['spacing'] *= LEN_CONV['Angstrom']['Bohr']
    # for key in conf['PP'] :
        # conf['PP'][key.capitalize()] = conf['PP'][key]

    return conf
