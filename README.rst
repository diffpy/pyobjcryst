.. image:: https://travis-ci.org/diffpy/pyobjcryst.svg?branch=master
   :target: https://travis-ci.org/diffpy/pyobjcryst

.. image:: https://codecov.io/gh/diffpy/pyobjcryst/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/diffpy/pyobjcryst

pyobjcryst
==========

Python bindings to ObjCryst++, the Object-Oriented Crystallographic Library.

The documentation for this release of pyobjcryst can be found on-line at
http://diffpy.github.io/pyobjcryst.


REQUIREMENTS
------------

pyobjcryst requires Python 3.7, 3.6, 3.5 or 2.7, C++ compiler and
the following software:

* ``libobjcryst`` - Object-Oriented Crystallographic Library for C++,
  https://github.com/diffpy/libobjcryst
* ``setuptools``  - tools for installing Python packages
* ``NumPy`` - library for scientific computing with Python
* ``python-dev`` - header files for interfacing Python with C
* ``libboost-all-dev`` - Boost C++ libraries and development files
* ``scons`` - software construction tool (optional)

We recommend to use `Anaconda Python <https://www.anaconda.com/download>`_
as it allows to install all software dependencies together with
pyobjcryst.  For other Python distributions it is necessary to
install the required software separately.  As an example, on Ubuntu
Linux the required software can be installed using ::

   sudo apt-get install \
      python-setuptools python-numpy scons \
      build-essential python-dev libboost-all-dev


INSTALLATION
------------

The preferred method is to use Anaconda Python and install from the
"diffpy" channel of Anaconda packages ::

   conda config --add channels diffpy
   conda install pyobjcryst

pyobjcryst is also included in the "diffpy-cmi" collection
of packages for structure analysis ::

   conda install diffpy-cmi

If you prefer to use other Python distribution or install from sources,
you must first install the libobjcryst library as per the instructions at
https://github.com/diffpy/libobjcryst.  Make sure other required
software is also in place and then run::

   python setup.py install

You may need to use ``sudo`` with system Python so the process is
allowed to copy files to system directories.  If administrator (root)
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


DEVELOPMENT
-----------

pyobjcryst is an open-source software developed as a part of the
DiffPy-CMI complex modeling initiative at the Brookhaven National
Laboratory.  The pyobjcryst sources are hosted at
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
   P="$(conda info --json | grep default_prefix | cut -d\" -f4)"
   export CPATH=$P/include
   export LIBRARY_PATH=$P/lib
   export LDFLAGS=-Wl,-rpath,$P/lib
   # compile and re-install pyobjcryst
   scons -j4 build=debug develop

Note the Anaconda package for the required libobjcryst library is built
with a C++ compiler provided by Anaconda.  This may cause incompatibility
with system C++.  In such case please use Anaconda C++ to build pyobjcryst.


CONTACTS
--------

For more information on pyobjcryst please visit the project web-page

http://www.diffpy.org

or email Prof. Simon Billinge at sb2896@columbia.edu.
