.. _download_and_install:

============
Installation
============

Requirements
============

* Python_ 3.6 or newer
* NumPy_ 1.8.0 or newer
* SciPy_ 0.10 or newer

Optional:

* pylibxc_ (for using exchange-correlation functionals other than LDA)
* pyFFTW_  (for Fast Fourier Transform)
* ASE_  (for dealing with structures, dynamics, and more)
* xmltodict_ (For UPF pseudopotential)
* upf_to_json_ (For UPF pseudopotential)
* mpi4py_ (MPI for python)
* mpi4py-fft_ (Fast Fourier Transforms with MPI)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _xmltodict: https://github.com/martinblech/xmltodict
.. _upf_to_json: https://github.com/simonpintarelli/upf_to_json


Installation from source
========================

You can get the source from `gitlab <https://gitlab.com/pavanello-research-group/dftpy.git>`_.

Git clone:
----------

    $ git clone https://gitlab.com/pavanello-research-group/dftpy.git


Environment
^^^^^^^^^^^

    Add ``dftpy/src`` to your `PYTHONPATH` environment variable.

Installation
^^^^^^^^^^^^

    Alternatively, you can install ``DFTpy`` with ``python -m pip install .``

    You also can install all the optional packages with ``python -m pip install .[all]``


.. note::

    Because ``DFTpy`` still under active development, non-backward-compatible changes can happen at any time. Please, clone the lastest release often.
