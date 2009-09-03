#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools import Extension
import fix_setuptools_chmod
import glob

module = Extension('pyobjcryst._pyobjcryst', 
        glob.glob("boost/*.cpp"),
        include_dirs = ['include/ObjCryst', 'boost'],
        library_dirs = ["lib"],
        libraries = ["objcryst", "boost_python"],
        extra_compile_args = ["-DREAL=double"]
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

        # danse.deploy extends setuptools in order to support scons setups, and
        # various other features.  These will allow us to use the OpenAlea
        # build system without considerable effort.
        setup_requires = ['danse.deploy'],
        dependency_links = ['http://dev.danse.us/packages'],

        # This tells danse.deploy where to put and find the shared libraries
        # modules.
        lib_dirs = { 'lib' : 'lib' },
        # And the library header files
        inc_dirs = { 'include' : 'include' },
        # And the extension module
        #package_data = {'' : ['_pyobjcryst.pyd', '_pyobjcryst.so'],},

        # This is a must, since the shared libraries are in the egg.
        zip_safe = False,

        # Now we can tell openalea.deploy where to find our scons file
        scons_scripts=['SConstruct'],

)

# End of file
