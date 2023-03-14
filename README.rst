.. image:: https://travis-ci.org/diffpy/pyobjcryst.svg?branch=master
   :target: https://travis-ci.org/diffpy/pyobjcryst

.. image:: https://codecov.io/gh/diffpy/pyobjcryst/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/diffpy/pyobjcryst

pyobjcryst
==========

Python bindings to ObjCryst++, the Object-Oriented Crystallographic Library.

The documentation for this release of pyobjcryst can be found on-line at
https://pyobjcryst.readthedocs.io/


INSTALLATION
------------
pyobjcryst is available for Python 3.7 (deprecated), and 3.8 to 3.11.

Using conda (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend to use **conda** as it allows to install all software dependencies
together with pyobjcryst, without too much compiling hastle.

Two distributions can be used:

* `Anaconda Python <https://www.anaconda.com/download>`_, the historical
  main conda-based distribution
* `Mamba-forge <https://github.com/conda-forge/miniforge/releases>`_ , which
  has the advantage off providing **mamba** in addition to conda, and is
  *much faster* when resolving dependencies during installation. It also
  uses by default the conda-forge repository, which is what almost all
  users would want.

Using conda, we you can install pyobjcryst using the "conda-forge" channel ::

   conda install -c conda-forge pyobjcryst

Alternatively using mamba ::

   mamba install pyobjcryst

You can also install from the "diffpy" channel - especially if you use
pyobjcryst allong with the other diffpy tools for PDF calculations,
but it is not updated as often as conda-forge.

pyobjcryst is also included in the "diffpy-cmi" collection
of packages for structure analysis ::

   conda install -c diffpy diffpy-cmi

Complete new conda environment with optional GUI and jupyter dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you have installed `Mamba-forge <https://github.com/conda-forge/miniforge/releases>`_,
you can directly create a new conda environment named `pyobjcryst` with all useful dependencies (including
jupyter-lab, python 3.11..) using ::

   mamba create -n pyobjcryst python=3.11 pyobjcryst matplotlib py3dmol jupyterlab ipympl multiprocess

Then you can activate the environment and launch jupyter-lab using ::

   conda activate pyobjcryst
   jupyter-lab

From source
^^^^^^^^^^^
The requirements are:

* ``libobjcryst`` - Object-Oriented Crystallographic Library for C++,
  https://github.com/diffpy/libobjcryst
* ``setuptools``  - tools for installing Python packages
* ``NumPy`` - library for scientific computing with Python
* ``python-dev`` - header files for interfacing Python with C
* ``libboost-all-dev`` - Boost C++ libraries and development files
* ``scons`` - software construction tool (optional)

The above requirements are easily installed through conda using e.g.::

  conda install numpy compilers boost scons libobjcryst

Alternatively, on Ubuntu Linux the required software can be installed using::

   sudo apt-get install \
      python-setuptools python-numpy scons \
      build-essential python-dev libboost-all-dev


The libobjcryst library can also be installed as per the instructions at
https://github.com/diffpy/libobjcryst. Make sure other required
software are also in place and then run from the pyobjcryst directory::

   pip install .

You may need to use ``sudo`` with system Python so the process is
allowed to copy files to system directories, unless you are installing
into a conda environment.  If administrator (root)
access is not available, see the usage information from
``python setup.py install --help`` for options to install to
a user-writable location.  The installation integrity can be
verified by executing the included tests with ::

   python -m pyobjcryst.tests.run

An alternative way of installing pyobjcryst is to use the SCons tool,
which can speed up the process by compiling C++ files in several
parallel jobs (-j4)::

   scons -j4 install

See ``scons -h`` for description of build targets and options.

Optional graphical dependencies for jupyter notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some of the classes can produce graphical outputs, which can be
displayed in a jupyter notebook:

* a Crystal structure can be displayed in 3D: this requires the
  ``py3dmol`` and ``ipywidgets`` modules. See the notebook
  ``examples/cystal_3d_widget.ipynb``
* a PowderPattern can be displayed (and live-updated) if
  ``matplotlib`` and ``ipympl`` are installed. See the
  notebook ``examples/cimetidine-structure-solution-powder.ipynb``

In short, ``conda install jupyter matplotlib ipympl py3dmol``
will give you all the required dependencies. You can also
use this in jupyterlab.

These packages can also be installed using ``pip`` if you do not use conda.

DEVELOPMENT
-----------

pyobjcryst is an open-source software originally developed as a part of the
DiffPy-CMI complex modeling initiative at the Brookhaven National
Laboratory, and is also further developed at ESRF.
The pyobjcryst sources are hosted at
https://github.com/diffpy/pyobjcryst.

Feel free to fork the project and contribute.  To install pyobjcryst
in a development mode, where its sources are directly used by Python
rather than copied to a system directory, use ::

   python setup.py develop --user

When developing it is preferable to compile the C++ files with
SCons using the ``build=develop`` option, which compiles the extension
module with debug information and C-assertions checks ::

   scons -j4 build=debug develop

The build script checks for a presence of ``sconsvars.py`` file, which
can be used to permanently set the ``build`` variable.  The SCons
construction environment can be further customized in a ``sconscript.local``
script.  The package integrity can be verified by executing unit tests with
``scons -j4 test``.

When developing with Anaconda Python it is essential to specify
header path, library path and runtime library path for the active
Anaconda environment.  This can be achieved by setting the ``CPATH``,
``LIBRARY_PATH`` and ``LDFLAGS`` environment variables as follows::

   # resolve the prefix directory P of the active Anaconda environment
   P=$CONDA_PREFIX
   export CPATH=$P/include
   export LIBRARY_PATH=$P/lib
   export LDFLAGS=-Wl,-rpath,$P/lib
   # compile and re-install pyobjcryst
   scons -j4 build=debug develop

Note the Anaconda package for the required libobjcryst library is built
with a C++ compiler provided by Anaconda.  This may cause incompatibility
with system C++.  In such case please use Anaconda C++ to build pyobjcryst.

Quick conda environment from libobjcryst and pyobjcryst sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``conda`` is available, you can create a pyobjcryst environment
from the git repositories (downloaded in the current directory) using::

  conda create --yes --name pyobjcryst numpy matplotlib ipywidgets jupyter
  conda install --yes  -n pyobjcryst -c conda-forge boost scons py3dmol
  conda activate pyobjcryst
  git clone https://github.com/diffpy/libobjcryst.git
  cd libobjcryst
  scons -j4 install prefix=$CONDA_PREFIX
  cd ..
  git clone https://github.com/diffpy/pyobjcryst.git
  cd pyobjcryst
  export CPATH=$CONDA_PREFIX/include
  export LIBRARY_PATH=$CONDA_PREFIX/lib
  export LDFLAGS=-Wl,-rpath,$CONDA_PREFIX/lib
  scons -j4 install prefix=$CONDA_PREFIX


CONTACTS
--------

For more information on pyobjcryst please visit the project web-page

http://www.diffpy.org

or email Prof. Simon Billinge at sb2896@columbia.edu.

You can also contact Vincent Favre-Nicolin (favre@esrf.fr) if you
are using pyobjcryst outside diffpy, e.g. to display structures
in a notebook, refine powder patterns or solve structures using the
global optimisation algorithms, etc..
