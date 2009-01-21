#!/usr/bin/env python


from setuptools import setup, find_packages
import fix_setuptools_chmod

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
)

# End of file
