#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Tests for crystal module."""

import unittest

from pyobjcryst.tests.pyobjcrysttestutils import makeC60

from pyobjcryst.refinableobj import RefinableObjClock



class TestClocks(unittest.TestCase):

    def testClockIncrement(self):
        """Make sure that clocks increment properly."""
        c = makeC60()
        m = c.GetScatterer("c60")

        ref = RefinableObjClock()
        mclock = m.GetClockScatterer()

        self.assertTrue( mclock > ref )
        ref.Click()
        self.assertFalse( mclock > ref )

        m[0].X = 0.01
        self.assertTrue( mclock > ref )
        ref.Click()
        self.assertFalse( mclock > ref )

        m[1].X = 0.01
        self.assertTrue( mclock > ref )
        ref.Click()
        self.assertFalse( mclock > ref )

        m[1].Y = 0.01
        self.assertTrue( mclock > ref )
        ref.Click()
        self.assertFalse( mclock > ref )

        m.Q0 = 1.001
        self.assertTrue( mclock > ref )
        ref.Click()
        self.assertFalse( mclock > ref )


if __name__ == "__main__":
    unittest.main()
