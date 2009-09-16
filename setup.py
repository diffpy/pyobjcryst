#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools import Extension
import fix_setuptools_chmod
import sys
import glob

# Include directories
from numpy.distutils.misc_util import get_numpy_include_dirs
include_dirs = get_numpy_include_dirs()
include_dirs.extend(['include/ObjCryst'])

# Compiler args

# Define the extension
module = Extension('pyobjcryst._pyobjcryst', 
        glob.glob("extensions/*.cpp"),
        include_dirs = include_dirs,
        library_dirs = ["lib"],
        libraries = ["objcryst", "boost_python"],
        define_macros = [("REAL","double")]
        )

# define distribution
dist =  setup(
        name = "pyobjcryst",
        version = "0.1a1",
        author = "Christopher L. Farrow",
        author_email = "clf2121@columbia.edu",
        description = "Bindings of ObjCryst++ into python",
        license = "BSD",
        url = "http://www.diffpy.org/",

        # What we're installing
        packages = ['pyobjcryst'],
        ext_modules = [module],
        scripts = ['applications/pyobjcryst-config'],

        # danse.deploy extends setuptools in order to support scons setups, and
        # various other features.
        setup_requires = ['danse.deploy'],
        dependency_links = ['http://dev.danse.us/packages'],

        # This tells danse.deploy where to put and find the shared libraries.
        lib_dirs = { 'lib' : 'lib' },
        # And the library header files
        inc_dirs = { 'include/ObjCryst' : 'include/ObjCryst' },

        # This is a must, since the shared libraries are in the egg.
        zip_safe = False,

        # Now we can tell danse.deploy where to find our scons file
        scons_scripts=['SConstruct'],

)

# End of file
