import numpy as np
from numpy.lib import format as npyf
import sys
import types

from dftpy.mpi import sprint


def write(fname, data, grid = None, single = False, version = (1, 0), close = True, datarep = 'native'):
    """
    Write npy file

    Args:
        fname: Name of output file or a file descriptor
        data: the value to output
        single: parallel version but run one processor
        version: the version of npy format
        close: close the file or not at the end
        datarep: Data representation

    Raises:
        AttributeError: Not support Fortran order

    Notes:
        For datarep, ``native`` fastest, but ``external32`` most portable
    """
    if single :
        return np.save(fname, data)
    if hasattr(data, 'grid'):
        grid = data.grid
    # serial version use numpy.save
    if grid is None or grid.mp.size == 1 :
        return np.save(fname, data)

    mp = grid.mp
    MPI = mp.MPI
    npyf._check_version(version)

    if isinstance(fname, str):
        fh = MPI.File.Open(mp.comm, fname, amode = MPI.MODE_CREATE | MPI.MODE_WRONLY)
    else :
        fh = fname

    if single :
        return _write_single(fh, data, mp, close)

    fp = _write_header(fh, data, version = version, grid = grid)
    _write_value(fh, data, fp = fp, grid = grid, datarep = datarep)
    if close :
        fh.Close()

def _write_single(fh, data, mp, version = (1, 0), close = True):
    if hasattr(fh, 'read'):
        return np.save(fh, data)
    _write_header_single(fh, data, version)
    _write_value_single(fh, data)

def _write_header_single(fh, data, version = (1, 0), grid = None):
    if hasattr(data, 'grid'):
        grid = data.grid
    header = npyf.header_data_from_array_1_0(data)
    if header['fortran_order'] :
        raise AttributeError("Not support Fortran order")
    if grid.mp.rank == 0:
        header['shape'] = grid.nrR
        _write_array_header_single(fh, header, version)
        fp = fh.Get_position()
    else:
        fp = 0
    grid.mp.comm.Bcast(fp, root = 0)
    return fp

def _write_header(fh, data, version = (1, 0), grid = None):
    if hasattr(data, 'grid'):
        grid = data.grid
    header = npyf.header_data_from_array_1_0(data)
    if header['fortran_order'] :
        raise AttributeError("Not support Fortran order")
    if grid.mp.rank == 0:
        header['shape'] = grid.nrR
        _write_array_header_single(fh, header, version)
        fp = fh.Get_position()
    else:
        fp = 0
    grid.mp.comm.Bcast(fp, root = 0)
    return fp

def _write_array_header_single(fh, header, version = (1, 0)):
    print('hhhh1', header)
    fstr = ["{"]
    for key, value in sorted(header.items()):
        fstr.append("'%s': %s, " % (key, repr(value)))
    fstr.append("}")
    fstr = "".join(fstr)
    fstr = npyf._filter_header(fstr)
    fstr = npyf._wrap_header(fstr, version)
    print('hhhh2', fstr)
    fh.Write(fstr)

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

def read(fname, data=None, grid=None, single=False, close = True, datarep = 'native'):
    """
    Read npy file

    Args:
        fname: Name of output file or a file descriptor
        data: stored the read data
        single: parallel version but run one processor
        grid: grid of the field
        close: close the file or not at the end
        datarep: Data representation

    Raises:
        AttributeError: Not support Fortran order

    """
    if hasattr(data, 'grid'):
        grid = data.grid
    # No grid and serial version use numpy.load
    if grid is None or grid.mp.size == 1 :
        return np.load(fname)

    MPI = grid.mp.MPI
    if isinstance(fname, str):
        fh = MPI.File.Open(grid.mp.comm, fname, amode=MPI.MODE_RDONLY)
    else :
        fh = fname

    if single :
        return _read_single(fh, data, close)

    shape, fortran_order, dtype = _read_header(fh)
    if 'fortran_order' :
        raise AttributeError("Not support Fortran order")
    if not(np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
        raise AttributeError("The shape is not match with grid")
    if data is None :
        data = np.empty(grid.nr, dtype=dtype, order='C')

    data=_read_value(fh, data, grid=grid, datarep=datarep)
    if close :
        fh.Close()
    return data

def _read_single(fh, data=None, close = True):
    if hasattr(fh, 'read'):
        return np.load(fh)
    shape, fortran_order, dtype = _read_header(fh)
    if data is None :
        if fortran_order :
            order = 'F'
        else :
            order = 'C'
        data = np.empty(shape, dtype=dtype, order=order)
    data=_read_value_single(fh, data)
    if close :
        fh.Close()
    return data

def _read_header(fh):
    if not hasattr(fh, 'read'):
        return _read_header_single(fh)
    version = npyf.read_magic(fh)
    npyf._check_version(version)
    return npyf._read_array_header(fh, version)

def _read_header_single(fh):
    version = _read_magic_single(fh)
    npyf._check_version(version)
    return _read_array_header_single(fh, version)

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

def _read_byte_single(fh, n, *args, **kwargs):
    data = bytearray(n)
    fh.Read(data)
    return data

def _read_byte_single_1(fh, n, *args, **kwargs):
    data = np.empty(n, dtype = np.byte)
    fh.Read(data)
    return data.tobytes()

def _read_value_single(fh, data):
    fh.Read(data)
    return data

def _read_magic_single(fh, magic=None):
    magic_str = _read_byte_single(fh, npyf.MAGIC_LEN)
    if magic_str[:-2] != npyf.MAGIC_PREFIX and magic_str[:-2] != magic :
        raise ValueError("Wrong magic string", magic_str)
    major, minor = magic_str[-2:]
    print(major, 'kkkkk', minor)
    return (major, minor)

def _read_array_header_single(fh, version):
    """
    same as numpy.lib.format._read_array_header
    """
    import struct
    hinfo = npyf._header_size_info.get(version)
    hlength_type, encoding = hinfo

    hlength_str = _read_byte_single(fh, struct.calcsize(hlength_type))
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    header = _read_byte_single(fh, header_length)
    header = header.decode(encoding)

    header = npyf._filter_header(header)
    d = npyf.safe_eval(header)
    dtype = npyf.descr_to_dtype(d['descr'])

    return d['shape'], d['fortran_order'], dtype
