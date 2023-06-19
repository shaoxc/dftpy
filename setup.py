#!/usr/bin/env python3
from setuptools import setup, find_packages
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


with open('src/dftpy/__init__.py') as fd :
    lines = fd.read()
    __version__ = re.search('__version__ = "(.*)"', lines).group(1)
    __author__ = re.search('__author__ = "(.*)"', lines).group(1)
    __contact__ = re.search('__contact__ = "(.*)"', lines).group(1)
    __license__ = re.search('__license__ = "(.*)"', lines).group(1)

assert sys.version_info >= (3, 6)
description = "Python3 packages for Density Functional Theory"

with open('README.md') as fh :
    long_description = fh.read()

scripts = ['scripts/dftpy']
extras_require = {
        'libxc' : [
            'pylibxc2; python_version<"3.10"',
            # 'pylibxc @ git+https://gitlab.com/libxc/libxc.git;python_version>"3.9"',
            ],
        'upf' : ['upf_to_json'],
        'mpi': ['mpi4py', 'mpi4py-fft'],
        'all' : [
            'pylibxc2; python_version<"3.10"',
            # 'pylibxc @ git+https://gitlab.com/libxc/libxc.git;python_version>"3.9"',
            'upf_to_json',
            'mpi4py',
            'mpi4py-fft',
            'pyfftw',
            ],
        }

release = 0
if release :
    VERSION = {'version' : __version__}
else :
    VERSION = {
            'use_scm_version': {'version_scheme': 'post-release'},
            'setup_requires': [
                'setuptools_scm',
                'importlib-metadata>=0.12;python_version<"3.8"'],
            }

setup(name='dftpy',
      description=description,
      long_description=long_description,
      url='http://dftpy.rutgers.edu',
      author=__author__,
      author_email=__contact__,
      license=__license__,
      **VERSION,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      packages=find_packages('src'),
      package_dir={'':'src'},
      scripts=scripts,
      include_package_data=True,
      extras_require = extras_require,
      install_requires= parse_requirements())
