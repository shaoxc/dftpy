def view_density(mol,level=2.0E-2):
    '''
    Visualize a DFTpy system on jupyter notebooks.
    Uses ipyvolume to render the density.
    '''
    from dftpy.system import System
    import numpy as np
    if not isinstance(mol,System):
        raise AttributeError("argument must be an instance of DFTpy system")
    try:
        import ipyvolume as ipv
    except:
        raise ModuleNotFoundError("Must install ipyvolume")
    rho = np.reshape(mol.field,np.shape(mol.field)[:3])
    density = np.pad(rho, [[0,1],[0,1],[0,1]], mode="wrap")
    ipv.figure()
    ipv.plot_isosurface(density,level)
    ipv.show()


def view_ions(mol):
    '''
    Visualize a DFTpy system on jupyter notebooks.
    Uses ipyvolume to render the ionic positions.
    '''
    from dftpy.system import System
    import numpy as np
    if not isinstance(mol,System):
        raise AttributeError("argument must be an instance of DFTpy system")
    try:
        import ipyvolume as ipv
    except:
        raise ModuleNotFoundError("Must install ipyvolume")
    pos = mol.ions.pos.to_crys()
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
    
    ipv.figure()
    ipv.scatter(val[:,0],val[:,1],val[:,2], marker='sphere', size=8,color='blue')
    ipv.xyzlim(0-tol2, 1+tol2)
    ipv.show()

