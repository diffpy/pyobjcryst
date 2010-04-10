#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools import Extension
import fix_setuptools_chmod
import sys
import glob

# Include directories
from numpy.distutils.misc_util import get_numpy_include_dirs
include_dirs = get_numpy_include_dirs()

# Compiler args

# Define the extension
module = Extension('pyobjcryst._pyobjcryst', 
        glob.glob("extensions/*.cpp"),
        include_dirs = include_dirs,
        libraries = ["ObjCryst", "boost_python-mt"],
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
        zip_safe = True,

)

# End of file
