#!/usr/bin/env python3
from setuptools import setup, find_packages
import subprocess
import sys
res = subprocess.run([sys.executable, 'tools/gitversion.py', '--write', 'src/dftpy/version.py'], capture_output=True, text=True)
version = res.stdout.split()[-1].strip()

setup(name='dftpy',
      version = version,
      packages=find_packages('src'),
      package_dir={'':'src'},
      include_package_data=True
      )
