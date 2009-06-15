#!/usr/bin/env python

from setuptools import setup, Extension
import fix_setuptools_chmod

# define distribution
dist =  setup(
        name = "pyobjcryst",
        version = "0.1a1",
        author = "Christopher L. Farrow",
        packages = ['pyobjcryst'],
        author_email = "clf2121@columbia.edu",
        description = "Bindings of ObjCryst++ into python",
        license = "BSD",
        url = "http://www.diffpy.org/",

        # OpenAlea.deploy extends setuptools in order to support scons setups,
        # and various other features.  These will allow us to use the OpenAlea
        # build system without considerable effort.
        setup_requires = ['openalea.deploy'],
        dependency_links = ['http://openalea.gforge.inria.fr/pi'],

        # This tells openalea.deploy where to put and find the shared libraries
        # modules.
        lib_dirs = { 'lib' : 'lib', 'pyobjcryst' : 'pyobjcryst' },
        # This is a must, since the shared libraries are linked from within the
        # egg.
        zip_safe = False,

        # Now we can tell distutils what to install
        scons_scripts=['SConstruct'],
)

# End of file
