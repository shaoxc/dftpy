import numpy as np
from dftpy.system import System
import importlib.util
try:
    import dftpy.visualize.ipv_viewer as ipvv
except Exception:
    import dftpy.visualize.mpl_viewer as mplv
import dftpy.visualize.mpl_viewer as mplv


def view_density(mol,level=None):
    '''
    Visualize a DFTpy system on jupyter notebooks.
    '''
    if not isinstance(mol,System):
        raise AttributeError("argument must be an instance of DFTpy system")
    density = mol.field
    rho = np.pad(density, [[0,1],[0,1],[0,1]], mode="wrap")
    isipv= importlib.util.find_spec("ipyvolume")
    isipv= False
    if isipv :
        ipvv.view_density(rho, level)
    else :
        mplv.plot_isosurface(rho, level)


def view_ions(mol, **kwargs):
    '''
    Visualize a DFTpy system on jupyter notebooks.
    '''
    if not isinstance(mol,System):
        raise AttributeError("argument must be an instance of DFTpy system")

    ions = mol.ions
    pos = ions.pos.to_crys()
    tol=1E-6
    tol2=2.0*tol
    ixyzA = np.mgrid[-1:2,-1:2,-1:2].reshape((3, -1)).T
    pbcpos=[]
    for p in ixyzA:
        pbcpos.append(pos+p)
    pbcpos=np.asarray(pbcpos).reshape((-1,3))
    val=pbcpos
    for i in range(3):
        val=val[np.logical_and(val[:,i]<1+tol, val[:,i]>-tol)]

    isipv= importlib.util.find_spec("ipyvolume")
    isipv= False
    if isipv :
        ipvv.view_ions(val, tol2)
    else :
        mplv.plot_scatter(val, tol2)
