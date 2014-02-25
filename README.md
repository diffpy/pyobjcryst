#PyObjCryst

Python bindings to ObjCryst++ Object-Oriented Crystallographic Library

The documentation for this release of PyObjCryst can be found on-line at
    
    http://diffpy.github.io/doc/pyobjcryst/

## REQUIREMENTS

The diffpy.srreal requires Python 2.6 or 2.7 and the following software:

    setuptools   -- tools for installing Python packages
    scons        -- software constructions tool (1.0 or later)
    python-dev   -- header files for interfacing Python with C
    libboost-dev -- Boost C++ libraries development files (1.43 or later)
    ObjCryst++   -- Object-Oriented Crystallographic Library
    
Some of the required software may be available in the system package manager,
for example, on Ubuntu Linux the dependencies can be installed as:

    sudo apt-get install \
        python-setuptools scons build-essential python-dev libboost-dev
        
For Mac OS X machine with the MacPorts package manager one could do

    sudo port install \
        python27 py27-setuptools scons boost

When installing with MacPorts, make sure the MacPorts bin directory is the
first in the system PATH and that python27 is selected as the default
Python version in MacPorts:

    sudo port select --set python python27

For other required packages see their respective web pages for installation
instructions.

## INSTALLATION

The easiest option is to use the latest DiffPy-CMI release bundle from
http://www.diffpy.org/, which comes with pyobjcryst and all other
dependencies included.

If you prefer to install from sources:

Installing ObjCryst++

We provide a SCons build script and ObjCryst++ source bundle to make it easier
to build ObjCryst++ and its dependencies (cctbx and newmat) as a shared
library. This requires SCons (http://www.scons.org) to be installed on your
computer. Here's how to install.

 1. Download ObjCryst-latest.tar.gz from http://dev.danse.us/packages/ to the
 directory containing INSTALL.txt ::

    > wget http://dev.danse.us/packages/ObjCryst-latest.tar.gz

 2. Extract the archive to the libobjcryst directory ::

    > tar xzvf ObjCryst-latest.tar.gz -C libobjcryst

 3. From the libobjcryst directory run the following command ::

    > scons install

    This will build and install the shared libraries, and header files to
    standard system-dependent locations.  Run `scons -h` for other installation
    options.

This build method has been tested on Linux platforms with recent GNU and Intel
C++ compilers.

Once you have done this, you can install PyObjCryst as instructed below.

## Installing PyObjCryst

To install PyObjCryst, you must have ObjCryst++ installed as a shared library
(see above). Once this is done, type the following from the command line from
the directory containing README.txt ::

> python setup.py install

For installation options, type ::

> python setup.py --help install


## DEVELOPMENT

pyobjcryst is an open-source software developed as a part of the
DiffPy-CMI complex modeling initiative at the Brookhaven National
Laboratory.  The pyobjcryst sources are hosted at

    https://github.com/diffpy/pyobjcryst

Feel free to fork the project and contribute.  To install pyobjcryst
in a development mode, where the sources are directly used by Python
rather than copied to a system directory, use

    python setup.py develop --user


## CONTACTS

For more information on diffpy.srreal please visit the project web-page

    http://www.diffpy.org/

or email Prof. Simon Billinge at sb2896@columbia.edu.
