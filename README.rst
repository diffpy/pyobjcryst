|Icon| |title|_
===============

.. |title| replace:: pyobjcryst
.. _title: https://diffpy.github.io/pyobjcryst

.. |Icon| image:: https://avatars.githubusercontent.com/diffpy
        :target: https://diffpy.github.io/pyobjcryst
        :height: 100px

|PyPI| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/diffpy/pyobjcryst/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/diffpy/pyobjcryst/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/diffpy/pyobjcryst/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/diffpy/pyobjcryst

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/pyobjcryst
        :target: https://anaconda.org/conda-forge/pyobjcryst

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff
        :target: https://github.com/diffpy/pyobjcryst/pulls

.. |PyPI| image:: https://img.shields.io/pypi/v/pyobjcryst
        :target: https://pypi.org/project/pyobjcryst/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/pyobjcryst
        :target: https://pypi.org/project/pyobjcryst/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/diffpy/pyobjcryst/issues

Python bindings to ObjCryst++, the Object-Oriented Crystallographic Library.

For more information about the pyobjcryst library, please consult our `online documentation <https://diffpy.github.io/pyobjcryst>`_.

``pyobjcryst`` is an open-source software package originally developed as a part of the DiffPy-CMI
complex modeling initiative which originated in the DANSE project
at Columbia University. It was further developed at Brookhaven National Laboratory,
and Columbia University and the European Synchrotron Radiation Source (ESRF) and is now
maintained at Columbia and ESRF.
The pyobjcryst sources are hosted at https://github.com/diffpy/pyobjcryst.

Citation
--------

If you use diffpy.srfit in a scientific publication, we would like you to cite this package as


   P. Juhás, C. L. Farrow, X. Yang, K. R. Knox and S. J. L. Billinge,
   `Complex modeling: a strategy and software program for combining
   multiple information sources to solve ill posed structure and
   nanostructure inverse problems
   <http://dx.doi.org/10.1107/S2053273315014473>`__,
   *Acta Crystallogr. A* **71**, 562-568 (2015).

and

   V. Favre-Nicolin and R. Cerný,
   `FOX, 'free objects for crystallography': a modular approach to
   ab initio structure determination from powder diffraction
   <https://doi.org/10.1107/S0021889802015236>`__,
   *J. Appl. Cryst.*  **35**, 734-743 (2002)

The second paper describes the c++ crystallographic objects in
``ObjCryst++`` that are wrapped by ``pyobjcryst``

Installation
------------

The latest release of ``pyobjcryst`` runs in python versions 3.11, 3.12 and 3.13. You may
specify an earlier release if you need it to run in an earlier version of Python.

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
or `mamba <https://mamba.readthedocs.io/en/latest/>`__
and install from the "conda-forge" channel of Conda packages.
mamba works in the same way as conda but has the advantage of being much
faster when resolving dependencies during installation. It also uses by
default the conda-forge repository, which is what almost all users would want.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``pyobjcryst_env`` ::

        conda create -n pyobjcryst_env pyobjcryst
        conda activate pyobjcryst_env

To confirm that the installation was successful, type ::

        python -c "import pyobjcryst; print(pyobjcryst.__version__)"

The output should print the latest version displayed on the badges above.

To use mamba, replace ``conda`` with ``mamba`` in the commands above.

pyobjcryst is also included in the ``diffpy.cmi`` collection of packages for
structure analysis and so can be installed by ::

        conda install -c conda-forge diffpy.cmi

Optional graphical dependencies for jupyter notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some of the classes can produce graphical outputs, which can be
displayed in a jupyter notebook:

* a Crystal structure can be displayed in 3D: this requires the
  ``py3dmol`` and ``ipywidgets`` modules. See the notebook
  ``docs/examples/cystal_3d_widget.ipynb``
* a PowderPattern can be displayed (and live-updated) if
  ``matplotlib`` and ``ipympl`` are installed. See the
  notebook ``docs/examples/structure-solution-powder-cimetidine.ipynb``

Getting Started
---------------

You may consult our `online documentation <https://diffpy.github.io/pyobjcryst>`_ for tutorials and API references.

Alternative methods of installation
-----------------------------------

These approaches are not recommended but reproduced here for advanced users.
You can use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``pyobjcryst_env`` environment, type ::

        pip install pyobjcryst

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/diffpy/pyobjcryst/>`_. Once installed, ``cd`` into your ``pyobjcryst`` directory
and run the following ::

        pip install .

An alternative way of installing pyobjcryst is to use the SCons tool,
which can speed up the process by compiling C++ files in several
parallel jobs (-j4)::

        conda install scons
        conda install --file requirements/conda.txt
        scons -j4 dev

See ``scons -h`` for description of build targets and options.

Alternatively, on Ubuntu Linux the required software can be installed using ::

        sudo apt-get install \
             python-setuptools python-numpy scons \
             build-essential python-dev libboost-all-dev

If this doesn't work, please see the `requirements/conda.txt` file for the
latest list of requirements.

The ``libobjcryst`` library can also be installed as per the instructions at
https://github.com/diffpy/libobjcryst. Make sure other required software are
also in place and then run from the pyobjcryst directory ::

        pip install .

You may need to use sudo with system Python so the process is allowed to copy files to system
directories, unless you are installing into a conda environment. If administrator (root) access is not
available, see the usage information from python setup.py install --help for options to install
to a user-writable location.

Testing your installation
-------------------------

The installation integrity can be verified by executing the included tests with

First install test dependencies then type pytest::

        conda install --file requirements/tests.txt
        pytest


Support and Contribute
----------------------

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/diffpy/pyobjcryst/issues>`_ and/or `submit a fix as a PR <https://github.com/diffpy/pyobjcryst/pulls>`_.

Feel free to fork the project and contribute. To install pyobjcryst
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

When developing it is preferable to compile the C++ files with
SCons using the ``build=debug`` option, which compiles the extension
module with debug information and C-assertions checks ::

   scons -j4 build=debug dev

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/diffpy/pyobjcryst/blob/main/CODE-OF-CONDUCT.rst>`_.

Contact
-------

For more information on pyobjcryst please visit the project `web-page <https://diffpy.github.io/>`_ or email Simon Billinge at sb2896@columbia.edu.

You can also contact Vincent Favre-Nicolin (favre@esrf.fr) if you are using pyobjcryst outside diffpy, e.g. to display structures in a notebook, refine powder patterns or solve structures using the global optimisation algorithms, etc..

Acknowledgements
----------------

This package bundles the following IUCr data files for bona fide research use:

- **cpd-1a.prn:** Powder diffraction dataset from the `IUCr CPD Round Robin on Quantitative Phase Analysis <https://www.iucr.org/__data/iucr/powder/QARR/index.html>`_.

  Source: https://www.iucr.org/__data/iucr/powder/QARR/col/cpd-1a.prn

  Round Robin on Quantitative Phase Analysis: Madsen, I. (1997) ‘Round Robin on Quantitative Phase Analysis’, Powder Diffraction, 12(1), pp. 1–2. Available at: https://doi.org/10.1017/S0885715600020212.


``pyobjcryst`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.
