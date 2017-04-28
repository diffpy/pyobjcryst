#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Test the converters.

This verifies results from tests built into the _registerconverters module.
"""

import unittest
from pyobjcryst._pyobjcryst import getTestVector, getTestMatrix
import numpy

class TestConverters(unittest.TestCase):

    def testVector(self):
        tv = numpy.arange(3, dtype=float)
        v = getTestVector()
        self.assertTrue( numpy.array_equal(tv, v) )
        return

    def testMatrix(self):
        tm = numpy.arange(6, dtype=float).reshape(3, 2)
        m = getTestMatrix()
        self.assertTrue( numpy.array_equal(tm, m) )
        return


if __name__ == "__main__":
    unittest.main()
