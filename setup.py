#!/usr/bin/env python3
from setuptools import setup, find_packages

def get_version(release=None):
    if release is None:
        with open('pyproject.toml', 'r') as fh:
            for line in fh:
                if line.startswith('version'):
                    release = True
                    break
                elif line.startswith('#version'):
                    release = False
                    break
    if release :
        VERSION = {'version' : None}
    else :
        VERSION = {
                'use_scm_version': {'version_scheme': 'post-release'},
                'setup_requires': ['setuptools_scm']
                }
    return VERSION
VERSION = get_version()

setup(name='dftpy',
      **VERSION,
      packages=find_packages('src'),
      package_dir={'':'src'},
      include_package_data=True
      )
