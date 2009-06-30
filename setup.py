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

        # This tells openalea.deploy where to put and find the shared
        # libraries. Shared libraries retrieved in this way will be installed
        # in the system's shared library location. Another location can be
        # chosen during install with the `--install-dyn-lib` option. 
        lib_dirs = { 'lib' : 'lib'},

        # This is a must, since the shared libraries are linked from within the
        # egg.
        zip_safe = False,

        # This tells openalea.deploy where to put and find binaries. Binaries
        # are installed locally within the egg, and not in a system directory.
        # Instead of binaries, we're using it to move the compiled modules into
        # the package directory so that they get installed with the python
        # files, and don't clutter the shared library directory.
        bin_dirs = { 'pyobjcryst' : 'pyobjcryst'},

        # Now we can tell openalea.deploy where to find our scons file
        scons_scripts=['SConstruct'],

)

# End of file
