.. _download_and_install:

============
Installation
============

Requirements
============

* Python_ 3.5 or newer
* NumPy_ 1.8.0 or newer
* SciPy_ 0.10 or newer

Optional:

* pylibxc_ (for different exchange-correlation)
* pyFFTW_  (for Fast Fourier Transform)
* ASE_  (for dealing with structures)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase


Installation from source
========================

You can get the source from github

Git clone:
----------

    $ git clone https://gitlab.com/pavanello-research-group/dftpy


:Environment:
    Add ``dftpy/src`` to your `PYTHONPATH` environment variable and add ``dftpy/scripts`` to `PATH`.

:Installation:
    Alternatively, you can install ``dftpy`` with ``python setup.py install --user``


.. note::

    Now, ``dftpy`` still under development, and big changes can happen at any time. So please follow the lastest release.
