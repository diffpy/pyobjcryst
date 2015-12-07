.. image:: https://travis-ci.org/diffpy/pyobjcryst.svg?branch=develop
   :target: https://travis-ci.org/diffpy/pyobjcryst

.. image:: http://codecov.io/github/diffpy/pyobjcryst/coverage.svg?branch=develop
   :target: http://codecov.io/github/diffpy/pyobjcryst?branch=develop

pyobjcryst
==========

Python bindings to ObjCryst++ Object-Oriented Crystallographic Library

The documentation for this release of pyobjcryst can be found on-line at
http://diffpy.github.io/pyobjcryst.


REQUIREMENTS
------------

pyobjcryst requires Python 2.6 or 2.7, C++ compiler and the following
software:

* ``libobjcryst`` - Object-Oriented Crystallographic Library for C++,
  https://github.com/diffpy/libobjcryst
* ``setuptools``  - tools for installing Python packages
* ``NumPy`` - library for scientific computing with Python
* ``python-dev`` - header files for interfacing Python with C
* ``libboost-all-dev`` - Boost C++ libraries and development files

With the exception of libobjcryst, the required software is commonly
available in system package manager, for example, on Ubuntu Linux the
required software can be installed as::

   sudo apt-get install \
      python-setuptools python-numpy scons \
      build-essential python-dev libboost-all-dev

For Mac OS X machine with the MacPorts package manager the installation is::

   sudo port install \
      python27 py27-setuptools py27-numpy scons boost

When installing with MacPorts, make sure that MacPorts bin directory is the
first in the system PATH and python27 is selected as the default Python
version in MacPorts::

   sudo port select --set python python27


INSTALLATION
------------

The easiest option is to use the latest DiffPy-CMI release bundle from
http://www.diffpy.org, which comes with pyobjcryst and all other
dependencies included.

If you prefer to install from sources, you must first install the libobjcryst
library as per the instructions at
https://github.com/diffpy/libobjcryst.  Make sure other required
software is in place as well and then run::

   sudo python setup.py install

This installs pyobjcryst for all users to the default system location.
If administrator (root) access is not available, see the usage info from
``python setup.py install --help`` for options for installing to a user-writable
location.  The installation integrity can be verified by changing to
the HOME directory and running::

   python -m pyobjcryst.tests.run

An alternative way of installing pyobjcryst is to use the SCons tool,
which can speed up the process by compiling C++ files in parallel (-j4)::

   sudo scons -j4 install

See ``scons -h`` for description of build targets and options for
choosing the installation directory.


DEVELOPMENT
-----------

pyobjcryst is an open-source software developed as a part of the
DiffPy-CMI complex modeling initiative at the Brookhaven National
Laboratory.  The pyobjcryst sources are hosted at
https://github.com/diffpy/pyobjcryst.

Feel free to fork the project and contribute.  To install pyobjcryst
in a development mode, where its sources are directly used by Python
rather than copied to a system directory, use::

   python setup.py develop --user

When developing it is preferable to compile the C++ files with
SCons using the ``build=develop`` option, which compiles the extension
module with debug information and C-assertions checks::

   scons -j4 build=debug develop

The build script checks for a presence of ``sconsvars.py`` file, which
can be used to permanently set the ``build`` variable.  The SCons
construction environment can be further customized in a ``sconscript.local``
script.  The package integrity can be verified by executing unit tests with
``scons -j4 test``.


CONTACTS
--------

For more information on pyobjcryst please visit the project web-page

http://www.diffpy.org

or email Prof. Simon Billinge at sb2896@columbia.edu.
