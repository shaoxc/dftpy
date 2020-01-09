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

* pylibxc_ (for using exchange-correlation functionals other than LDA)
* pyFFTW_  (for Fast Fourier Transform)
* ASE_  (for dealing with structures, dynamics, and more)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase


Installation from source
========================

You can get the source from gitlab

Git clone:
----------

    $ git clone https://gitlab.com/pavanello-research-group/dftpy


:Environment:
    Add ``dftpy/src`` to your `PYTHONPATH` environment variable and add ``dftpy/scripts`` to `PATH`.

:Installation:
    Alternatively, you can install ``DFTpy`` with ``python setup.py install --user``


.. note::

    Because ``DFTpy`` still under active development, non-backward-compatible changes can happen at any time. Please, clone the lastest release often.
