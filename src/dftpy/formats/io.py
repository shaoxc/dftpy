import os
from collections import namedtuple
from importlib import import_module
from dftpy.field import DirectField
from dftpy.mpi import SerialComm
from dftpy.ions import Ions

ioformat = namedtuple('ioformat', ['format', 'module', 'read', 'write', 'kind'])
iounkeys = ['mp', 'comm', 'kind', 'data_type', 'grid', 'names']

IOFormats= {
            "snpy"    : ioformat('snpy', 'dftpy.formats.snpy'  , 'read_snpy', 'write_snpy', ['ions', 'data', 'all']),
            "xsf"     : ioformat('xsf' , 'dftpy.formats.xsf'   , 'read_xsf' , 'write_xsf' , ['ions', 'data', 'all']),
            "pp"      : ioformat('qepp', 'dftpy.formats.qepp'  , 'read_qepp', 'write_qepp', ['ions', 'data', 'all']),
            "qepp"    : ioformat('qepp', 'dftpy.formats.qepp'  , 'read_qepp', 'write_qepp', ['ions', 'data', 'all']),
            "cube"    : ioformat('cube', 'dftpy.formats.cube'  , 'read_cube', 'write_cube', ['ions', 'data', 'all']),
            "den"     : ioformat('den' , 'dftpy.formats.den'   , 'read_den' , 'write_den' , ['data']),
            "xyz"     : ioformat('xyz' , 'dftpy.formats.xyz'   , 'read_xyz' , 'write_xyz' , ['ions']),
            "extxyz"  : ioformat('xyz' , 'dftpy.formats.xyz'   , 'read_xyz' , 'write_xyz' , ['ions']),
            "vasp"    : ioformat('vasp', 'dftpy.formats.vasp'  , 'read_vasp', 'write_vasp', ['ions']),
            "poscar"  : ioformat('vasp', 'dftpy.formats.vasp'  , 'read_vasp', 'write_vasp', ['ions']),
            "contcar" : ioformat('vasp', 'dftpy.formats.vasp'  , 'read_vasp', 'write_vasp', ['ions']),
            "ase"     : ioformat('ase' , 'dftpy.formats.ase_io', 'read_ase' , 'write_ase' , ['ions']),
            "pmg"     : ioformat('pmg' , 'dftpy.formats.pmg_io', 'read_pmg' , 'write_pmg' , ['ions']),
            "pymatgen": ioformat('pmg' , 'dftpy.formats.pmg_io', 'read_pmg' , 'write_pmg' , ['ions']),
        }

def guessType(infile, **kwargs):
    return guess_format(infile, **kwargs)

def guess_format(infile, **kwargs):
    basename = os.path.basename(infile).lower()
    ext = os.path.splitext(infile)[1].lower()
    format = IOFormats.get(basename, None)
    if format is None and len(ext)>1 :
        format = IOFormats.get(ext[1:], None)

    if format is None :
        format = ext[1:] if len(ext)>1 else basename
    else :
        format = format.format
    return format

def get_io_driver(infile, format = None, mode = 'r'):
    if format is None : format = guess_format(infile)
    iof = IOFormats.get(format, None)

    if iof is None :
        raise AttributeError("%s format not support yet" % format)

    try:
        module = import_module(iof.module)
        if 'r' in mode :
            func = getattr(module, iof.read, None)
            if not iof.read or func is None : raise AttributeError("%s format not support read" % iof.format)
            iof = iof._replace(read = func)
        if 'w' in mode or 'a' in mode :
            func = getattr(module, iof.write, None)
            if not iof.write or func is None : raise AttributeError("%s format not support write" % iof.format)
            iof = iof._replace(write = func)
    except Exception as e:
        raise e

    return iof

def read(infile, format=None, kind='ions', driver=None, **kwargs):
    if driver is None :
        driver = get_io_driver(infile, format, mode = 'r')
    elif isinstance(driver, str) :
        for key in iounkeys :
            if key in kwargs : kwargs.pop(key)
        driver = get_io_driver(infile, driver, mode = 'r')
    #
    if hasattr(driver, 'read') :
        values = driver.read(infile, format=format, **kwargs)
    else :
        raise AttributeError(f"Sorry, not support {driver} driver")
    #
    if kind == 'all' :
        if 'ions' not in driver.kind :
            values = (None, values, None)
        elif 'data' not in driver.kind :
            values = (values, None, None)
        elif len(values) == 2 :
            values = *values, None
        elif len(values) != 3 :
            raise AttributeError(f"Sorry, the {driver} driver should only return 3 values.")
    if kind == 'data' :
        if len(values) == 2 or len(values) == 3 :
            values = values[1]
    return values

def write(outfile, ions = None, data = None, information = None, format = None, comm = None, driver = None, **kwargs):
    if isinstance(data, Ions):
        if ions is None or isinstance(ions, DirectField):
            ions, data = data, ions
        else :
            raise AttributeError("Please check the input data")

    if comm is None :
        if hasattr(data, 'mp') :
            comm = data.mp.comm
        else :
            comm = SerialComm()
    if driver is not None :
        if comm.size > 1 and data is not None :
            data = data.gather()
        if isinstance(driver, str) :
            for key in iounkeys :
                if key in kwargs : kwargs.pop(key)
            driver = get_io_driver(outfile, driver, mode = 'w')
            if comm.rank == 0 : driver.write(outfile, ions = ions, data = data, **kwargs)
        elif hasattr(driver, 'write') :
            if comm.rank == 0 : driver.write(outfile, ions = ions, data = data, **kwargs)
        else :
            raise AttributeError(f"Sorry, not support {driver} driver")
    else :
        driver = get_io_driver(outfile, format, mode = 'w')
        if driver.format == "snpy": # only snpy format support MPI-IO
            return driver.write(outfile, ions=ions, data=data, information = information, **kwargs)
        else : # only rank==0 write
            if comm.size > 1 and data is not None :
                data = data.gather()
            if comm.rank == 0 : driver.write(outfile, ions = ions, data=data, information = information, **kwargs)

            comm.Barrier()
    return

def read_data(infile, format=None, **kwargs):
    return read(infile, format=format, kind='data', **kwargs)

def read_all(infile, format=None, **kwargs):
    return read(infile, format=format, kind='all', **kwargs)

def read_density(infile, format=None, **kwargs):
    kwargs.pop('data_type', None)
    return read(infile, format=format, kind='data', data_type='density', **kwargs)

def read_potential(infile, format=None, **kwargs):
    kwargs.pop('data_type', None)
    return read(infile, format=format, kind='data', data_type='potential', **kwargs)

def write_all(outfile, ions = None, data = None, information = None, format=None, **kwargs):
    return write(outfile, ions = None, data = None, information = None, format=format, **kwargs)

def write_density(outfile, data = None, ions = None, format = None, **kwargs):
    kwargs.pop('data_type', None)
    return write(outfile, data = data, ions = ions, format = format, data_type='density', **kwargs)

def write_potential(outfile, data = None, ions = None, format = None, **kwargs):
    kwargs.pop('data_type', None)
    return write(outfile, data = data, ions = ions, format = format, data_type='potential', **kwargs)
