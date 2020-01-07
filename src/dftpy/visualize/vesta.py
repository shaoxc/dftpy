def view_on_vesta(mol,saveas=None):
    '''
    View DFTpy system on VESTA. Must have VESTA installed.
    mol: DFTpy system
    saveas: string
    '''
    from dftpy.system import System
    from dftpy.formats.xsf import XSF
    import numpy as np
    import sys
    import os
    filexsf='./.view.xsf'
    if filexsf in os.listdir('./'):
        os.remove(filexsf)
    xsf = XSF(filexsf=filexsf)
    xsf.write(mol)
    print('OS is: ', sys.platform)
    if sys.platform == 'linux':
        os.system('VESTA '+filexsf)
    elif sys.platform == 'darwin': 
        os.system('open -a VESTA '+filexsf)
    else:
        raise Exception('Unknown OS')
    if saveas is not None:
        if isinstance(saveas,str):
            os.replace(filexsf,saveas)
        else:
            raise AttributeError("saveas must be a string")

