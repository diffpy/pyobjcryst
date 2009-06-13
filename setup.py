#!/usr/bin/env python


from setuptools import setup
import fix_setuptools_chmod

# OpenAlea.deploy extends setuptools in order to support scons setups, and
# various other features.
openalea_requires = ['openalea.deploy']
openalea_links = ['http://openalea.gforge.inria.fr/pi']

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

        # These will allow us to use the OpenAlea build system without any
        # effort.
        setup_requires = openalea_requires,
        dependency_links = openalea_requires,

        # Now we can tell distutils what to install
        scons_scripts=['SConstruct'],
        scons_parameters=['installpy'],
)

# End of file
