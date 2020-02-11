from dftpy.system import System
import numpy as np
try:
    import ipyvolume as ipv
except Exception:
    raise ModuleNotFoundError("Must install ipyvolume")


def view_density(density,level=None):
    '''
    Uses ipyvolume to render the density.
    '''
    if level is None :
        level=0.5*(np.max(density)+np.min(density))
    ipv.figure()
    ipv.plot_isosurface(density,level)
    ipv.show()


def view_ions(val, tol2 = 2E-6, color = None):
    '''
    Uses ipyvolume to render the ionic positions.
    '''
    ipv.figure()
    ipv.scatter(val[:,0],val[:,1],val[:,2], marker='sphere', size=8, color='blue')
    ipv.xyzlim(0-tol2, 1+tol2)
    ipv.show()
