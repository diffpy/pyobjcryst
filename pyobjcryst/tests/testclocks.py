#!/usr/bin/env python
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
