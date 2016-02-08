#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        Complex Modeling Initiative
#                   (c) 2013 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Unit tests for pyobjcryst.
"""


def testsuite():
    '''Build a unit tests suite for the pyobjcryst package.

    Return a unittest.TestSuite object.
    '''
    import unittest
    modulenames = '''
        pyobjcryst.tests.testcif
        pyobjcryst.tests.testclocks
        pyobjcryst.tests.testconverters
        pyobjcryst.tests.testcrystal
        pyobjcryst.tests.testmolecule
        pyobjcryst.tests.testrefinableobj
        pyobjcryst.tests.testutils
    '''.split()
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    mobj = None
    for mname in modulenames:
        exec ('import %s as mobj' % mname)
        suite.addTests(loader.loadTestsFromModule(mobj))
    return suite


def test():
    '''Execute all unit tests for the pyobjcryst package.
    Return a unittest TestResult object.
    '''
    import unittest
    suite = testsuite()
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result


# End of file
