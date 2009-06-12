#!/usr/bin/env python


from setuptools import setup, find_packages
import fix_setuptools_chmod

#So we can build scons stuff
import sys
setup_requires = ['openalea.deploy']
# web sites where to find eggs
dependency_links = ['http://openalea.gforge.inria.fr/pi']

# define distribution
dist = setup(
        name = "pyobjcryst",
        version = "0.1a1",
        author = "Christopher L. Farrow",
        packages = find_packages(),
        author_email = "clf2121@columbia.edu",
        description = "Bindings of ObjCryst++ into python",
        license = "BSD",
        url = "http://www.diffpy.org/",

        # openalea
        setup_requires = setup_requires,
        dependency_links = dependency_links,

        # scons
        scons_scripts=['SConstruct']
)

# End of file
