#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        Complex Modeling Initiative
#                   (c) 2019 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
"""Unit tests for pyobjcryst.spacegroup."""


import unittest

from pyobjcryst.spacegroup import SpaceGroup

# ----------------------------------------------------------------------------


class TestSpaceGroup(unittest.TestCase):

    def setUp(self):
        return

    def test___init__(self):
        "check SpaceGroup.__init__()"
        sg = SpaceGroup()
        self.assertEqual(1, sg.GetSpaceGroupNumber())
        self.assertRaises(ValueError, SpaceGroup, "invalid")
        sgfm3m = SpaceGroup("F m -3 m")
        self.assertEqual(225, sgfm3m.GetSpaceGroupNumber())
        sg3 = SpaceGroup("3")
        self.assertEqual(3, sg3.GetSpaceGroupNumber())
        return

    def test_ChangeSpaceGroup(self):
        "check SpaceGroup.ChangeSpaceGroup()"
        sg = SpaceGroup("F m -3 m")
        self.assertEqual("F m -3 m", sg.GetName())
        self.assertRaises(ValueError, sg.ChangeSpaceGroup, "invalid")
        self.assertEqual("F m -3 m", sg.GetName())
        sg.ChangeSpaceGroup("P1")
        self.assertEqual("P 1", sg.GetName())
        return


# End of class TestSpaceGroup

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
