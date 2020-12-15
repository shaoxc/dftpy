import numpy as np
from numpy.lib import format as npyf
# from dftpy.mpi import sprint


def write(fh, data, grid = None, single = False, version = (1, 0), datarep = 'native'):
    """
    Write npy file

    Args:
        fh: Name of output file or a file descriptor
        data: the value to output
        single: parallel version but run one processor
        version: the version of npy format, only support (1, 0), (2, 0), (3, 0)
        datarep: Data representation

    Raises:
        AttributeError: Not support Fortran order

    Notes:
        For datarep, ``native`` fastest, but ``external32`` most portable
    """
    if hasattr(data, 'grid'):
        grid = data.grid

    npyf._check_version(version)

    if single or (not hasattr(fh, 'Get_position')):
        return _write_single(fh, data, version)

    if grid.mp.rank == 0:
        fp = _write_header(fh, data, version = version, grid = grid)
        fp = fh.Get_position()
    else:
        fp = 0
    fp = grid.mp.comm.bcast(fp, root = 0)
    _write_value(fh, data, fp = fp, grid = grid, datarep = datarep)

def _write_single(fh, data, version = (1, 0)):
    if not hasattr(fh, 'Get_position'):
        # serial version use numpy.save
        return np.save(fh, data)
    _write_header(fh, data, version)
    _write_value_single(fh, data)

def _write_header(fh, data, version = (1, 0), grid = None):
    if hasattr(data, 'grid'):
        grid = data.grid
    header = npyf.header_data_from_array_1_0(data)
    if header['fortran_order'] :
        raise AttributeError("Not support Fortran order")
    if grid is None :
        shape = data.shape
    else :
        shape = grid.nrR
    header['shape'] = tuple(shape)
    npyf._write_array_header(fh, header, version)
    return

def _write_value_single(fh, data):
    fh.Write(data)
    return

def _write_value(fh, data, fp = None, grid=None, datarep = 'native'):
    if hasattr(data, 'grid'):
        grid = data.grid

    if fp is None :
        if grid.mp.rank == 0:
            fp = fh.Get_position()
        else:
            fp = 0
        grid.mp.comm.Bcast(fp, root = 0)
    MPI = grid.mp.MPI
    etype = MPI._typedict[data.dtype.char]
    filetype = etype.Create_subarray(grid.nrR, grid.nr, grid.offsets, order=MPI.ORDER_C)
    filetype.Commit()
    fh.Set_view(fp, etype, filetype, datarep=datarep)
    fh.Write_all(data)
    filetype.Free()
    return

def read(fh, data=None, grid=None, single=False, datarep = 'native'):
    """
    Read npy file

    Args:
        fh: Name of output file or a file descriptor
        data: stored the read data
        single: parallel version but run one processor
        grid: grid of the field
        datarep: Data representation

    Raises:
        AttributeError: Not support Fortran order

    """
    if single or (not hasattr(fh, 'Get_position')):
        return _read_single(fh, data)

    if hasattr(data, 'grid'):
        grid = data.grid

    shape, fortran_order, dtype = _read_header(fh)
    if 'fortran_order' :
        raise AttributeError("Not support Fortran order")
    if not(np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
        raise AttributeError("The shape is not match with grid")
    if data is None :
        data = np.empty(grid.nr, dtype=dtype, order='C')

    data=_read_value(fh, data, grid=grid, datarep=datarep)
    return data

def _read_single(fh, data=None):
    if not hasattr(fh, 'Get_position'):
        # serial version use numpy.save
        return np.load(fh)
    shape, fortran_order, dtype = _read_header(fh)
    if data is None :
        if fortran_order :
            order = 'F'
        else :
            order = 'C'
        data = np.empty(shape, dtype=dtype, order=order)
    data=_read_value_single(fh, data)
    return data

def _read_header(fh):
    version = npyf.read_magic(fh)
    npyf._check_version(version)
    return npyf._read_array_header(fh, version)

def _read_value(fh, data, grid=None, datarep = 'native'):
    if hasattr(data, 'grid'):
        grid = data.grid
    MPI = grid.mp.MPI
    fp = fh.Get_position()
    etype = MPI._typedict[data.dtype.char]
    filetype = etype.Create_subarray(grid.nrR, grid.nr, grid.offsets, order=MPI.ORDER_C)
    filetype.Commit()
    fh.Set_view(fp, etype, filetype, datarep=datarep)
    fh.Read_all(data)
    filetype.Free()
    return data

def _read_value_single(fh, data):
    fh.Read(data)
    return data
