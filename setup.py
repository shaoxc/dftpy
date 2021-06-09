#!/usr/bin/env python3
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import sys
import os
import re

def parse_requirements():
    requires = []
    with open('requirements.txt', 'r') as fr :
        for line in fr :
            pkg = line.strip()
            if pkg.startswith('git+'):
                pip_install_git(pkg)
            else:
                requires.append(pkg)
    return requires

def pip_install_git(link):
    os.system('pip install --upgrade {}'.format(link))
    return

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# from dftpy import __version__, __author__, __contact__, __license__
with open('src/dftpy/__init__.py') as fd :
    lines = fd.read()
    __version__ = re.search('__version__ = "(.*)"', lines).group(1)
    __author__ = re.search('__author__ = "(.*)"', lines).group(1)
    __contact__ = re.search('__contact__ = "(.*)"', lines).group(1)
    __license__ = re.search('__license__ = "(.*)"', lines).group(1)


assert sys.version_info >= (3, 6)
description = "DFTpy: A Python3 packages for Density Functional Theory",
long_description = """ `DFTpy` is an Density Functional Theory code based on a plane-wave
expansion of the electron density"""

scripts = ['scripts/dftpy']
extras_require = {
        'libxc' : ['pylibxc @ git+https://gitlab.com/libxc/libxc.git'],
        'upf' : ['xmltodict', 'upf_to_json'],
        'mpi4py': ['mpi4py @ git+https://bitbucket.org/mpi4py/mpi4py.git'],
        'all' : [
            'pylibxc @ git+https://gitlab.com/libxc/libxc.git',
            'ase>=3.21.1',
            'xmltodict',
            'upf_to_json',
            'mpi4py',
            'mpi4py-fft',
            'pyfftw',
            ],
        }

setup(name='dftpy',
      description=description,
      long_description=long_description,
      url='https://gitlab.com/pavanello-research-group/dftpy',
      version=__version__,
      author=__author__,
      author_email=__contact__,
      license=__license__,
      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      packages=find_packages('src'),
      package_dir={'':'src'},
      scripts=scripts,
      include_package_data=True,
      extras_require = extras_require,
      install_requires= parse_requirements())
