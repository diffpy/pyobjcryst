#!/usr/bin/env python

from setuptools import setup, find_packages
import fix_setuptools_chmod

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

        # openalea.deploy extends setuptools in order to support scons setups,
        # and various other features.  These will allow us to use the OpenAlea
        # build system without considerable effort.
        setup_requires = ['openalea.deploy'],
        dependency_links = ['http://openalea.gforge.inria.fr/pi'],

        # This tells openalea.deploy where to put and find the shared libraries
        # modules.
        lib_dirs = { 'lib' : 'lib' },
        # And the library header files
        inc_dirs = { 'include' : 'include' },
        package_data = {'' : ['_pyobjcryst.pyd', '_pyobjcryst.so'],},

        # This is a must, since the shared libraries are in the egg.
        zip_safe = False,

        # Now we can tell openalea.deploy where to find our scons file
        scons_scripts=['SConstruct'],

)

# End of file
