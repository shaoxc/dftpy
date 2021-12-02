import os
from collections import namedtuple
from importlib import import_module
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.field import DirectField
from dftpy.mpi import SerialComm

ioformat = namedtuple('ioformat', ['format', 'module', 'read', 'write', 'kind'])
iounkeys = ['mp', 'comm', 'kind', 'data_type', 'grid', 'names']

IOFormats= {
            "snpy"    : ioformat('snpy', 'dftpy.formats.snpy'  , 'read_snpy', 'write_snpy', ['cell', 'field', 'all']),
            "xsf"     : ioformat('xsf' , 'dftpy.formats.xsf'   , 'read_xsf' , 'write_xsf' , ['cell', 'field', 'all']),
            "pp"      : ioformat('qepp', 'dftpy.formats.qepp'  , 'read_qepp', 'write_qepp', ['cell', 'field', 'all']),
            "qepp"    : ioformat('qepp', 'dftpy.formats.qepp'  , 'read_qepp', 'write_qepp', ['cell', 'field', 'all']),
            "den"     : ioformat('den' , 'dftpy.formats.den'   , 'read_den' , 'write_den' , ['field']),
            "xyz"     : ioformat('xyz' , 'dftpy.formats.xyz'   , 'read_xyz' , 'write_xyz' , ['cell']),
            "extxyz"  : ioformat('xyz' , 'dftpy.formats.xyz'   , 'read_xyz' , 'write_xyz' , ['cell']),
            "vasp"    : ioformat('vasp', 'dftpy.formats.vasp'  , 'read_vasp', 'write_vasp', ['cell']),
            "poscar"  : ioformat('vasp', 'dftpy.formats.vasp'  , 'read_vasp', 'write_vasp', ['cell']),
            "contcar" : ioformat('vasp', 'dftpy.formats.vasp'  , 'read_vasp', 'write_vasp', ['cell']),
            "ase"     : ioformat('ase' , 'dftpy.formats.ase_io', 'read_ase' , 'write_ase' , ['cell']),
            "pmg"     : ioformat('pmg' , 'dftpy.formats.pmg_io', 'read_pmg' , 'write_pmg' , ['cell']),
            "pymatgen": ioformat('pmg' , 'dftpy.formats.pmg_io', 'read_pmg' , 'write_pmg' , ['cell']),
        }

def guessType(infile, **kwargs):
    basename = os.path.basename(infile)
    ext = os.path.splitext(infile)[1].lower()
    format = IOFormats.get(basename.lower(), None)
    if format is None : format = IOFormats.get(ext[1:], None)

    if format is None :
        raise AttributeError("%s not support yet" % infile)
    return format.format

def get_io_driver(infile, format = None, mode = 'r'):
    if format is None : format = guessType(infile)
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

def read_system(infile, format=None, driver=None, **kwargs):
    if driver is None :
        driver = get_io_driver(infile, format, mode = 'r')
        system = driver.read(infile, **kwargs)
    elif isinstance(driver, str) :
        for key in iounkeys :
            if key in kwargs : kwargs.pop(key)
        driver = get_io_driver(infile, driver, mode = 'r')
        system = driver.read(infile, **kwargs)
    elif hasattr(driver, 'read') :
        system = driver.read(infile, format=format, **kwargs)
    else :
        raise AttributeError(f"Sorry, not support {driver} driver")
    return system

def write_system(outfile, system, format = None, comm = None, driver = None, **kwargs):
    if comm is None :
        if hasattr(system.field, 'mp') :
            comm = system.field.mp.comm
        else :
            comm = SerialComm()
    if driver is not None :
        if isinstance(driver, str) :
            for key in iounkeys :
                if key in kwargs : kwargs.pop(key)
            driver = get_io_driver(outfile, driver, mode = 'w')
            driver.write(outfile, system, **kwargs)
        elif hasattr(driver, 'write') :
            if comm.rank == 0 : driver.write(outfile, system, format=format, **kwargs)
        else :
            raise AttributeError(f"Sorry, not support {driver} driver")
    else :
        driver = get_io_driver(outfile, format, mode = 'w')
        if driver.format == "snpy": # only snpy format support MPI-IO
            return driver.write(outfile, system, **kwargs)
        else : # only rank==0 write
            if comm.size > 1 and 'field' in driver.kind :
                total = system.field.gather()
                system = System(system.ions, name="DFTpy", field=total)

            if comm.rank == 0 : driver.write(outfile, system, **kwargs)

            comm.Barrier()
    return

def read_density(infile, format=None, **kwargs):
    struct = read_system(infile, format=format, **kwargs)
    return struct.field

def read(infile, format=None, **kwargs):
    struct = read_system(infile, format=format, **kwargs)
    kind = kwargs.get('kind', 'cell')
    if kind == 'cell' :
        return struct.ions
    elif kind == 'field' :
        return struct.field
    else :
        return struct

def write(outfile, data = None, ions = None, format=None, **kwargs):
    if isinstance(data, System):
        system = data
    elif isinstance(data, DirectField):
        system = System(ions, name="DFTpy", field=data)
    elif isinstance(data, Atom):
        if ions is None or isinstance(ions, DirectField):
            ions, data = data, ions
            system = System(ions, name="DFTpy", field=data)
        else :
            raise AttributeError("Please check the input data")
    else :
        raise AttributeError("Please check the input data")

    return write_system(outfile, system, format=format, **kwargs)
