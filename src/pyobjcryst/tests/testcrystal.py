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

"""Tests for crystal module."""

import unittest

from pyobjcryst.tests.pyobjcrysttestutils import (
    makeScatterer, makeCrystal, getScatterer, makeScattererAnisotropic)
from pyobjcryst.atom import Atom


class TestCrystal(unittest.TestCase):

    def testCrystalScope(self):
        """Test to see if the the crystal survives after it is out of scope."""
        sp, atom = makeScatterer()
        makeCrystal(sp, atom)

        # The crystal is out of scope. Since the lifetime of the atom and
        # scatterer are linked, the crystal should stay alive in memory.
        self.assertEqual("Ni", sp.GetName())
        self.assertEqual("Ni", atom.GetName())
        return

    def testScattererScope(self):
        """Test when atoms go out of scope before crystal."""
        sp2 = getScatterer()
        self.assertEqual("Ni", sp2.GetName())
        return

    def testScattererB(self):
        """Test Biso and Bij of scatterer."""
        sp1, junk = makeScatterer()
        self.assertTrue(sp1.IsIsotropic())
        sp2, junk = makeScattererAnisotropic()
        self.assertFalse(sp2.IsIsotropic())
        return

    def testNullData(self):
        """Make sure we get an error when trying to add or remove Null."""
        from pyobjcryst.crystal import Crystal
        c = Crystal()
        self.assertRaises(ValueError, c.AddScatterer, None)
        self.assertRaises(ValueError, c.RemoveScatterer, None)
        self.assertRaises(ValueError, c.AddScatteringPower, None)
        self.assertRaises(ValueError, c.RemoveScatteringPower, None)
        return

    def testRemoveFunctions(self):
        """Test the RemoveScatterer and RemoveScatteringPower method."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)

        # You can add scatterers with the same name. That should be a no-no.
        sp2, atom2 = makeScatterer()
        c.AddScatterer(atom2)
        c.AddScatteringPower(sp2)

        # These act according to the library. You can try to remove an object
        # that is not in the crystal, and it will gladly do nothing for you.

        # Remove the scatterers
        c.RemoveScatterer(atom)
        c.RemoveScatteringPower(sp)
        # Remove again
        c.RemoveScatterer(atom)
        c.RemoveScatteringPower(sp)

        # Try to remove scatterers that are not in the crystal
        c.RemoveScatterer(atom2)
        c.RemoveScatteringPower(sp2)
        return

    def testGetScatteringComponentList(self):
        """Test the RemoveScatterer and RemoveScatteringPower method."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)
        scl = c.GetScatteringComponentList()
        self.assertEqual(1, len(scl))

        sclcopy = scl[:]
        self.assertEqual(1, len(scl))

        del sclcopy[0]
        self.assertEqual(0, len(sclcopy))
        self.assertEqual(1, len(scl))

        del scl[0]
        self.assertEqual(0, len(scl))

        return

    def testGetScatterer(self):
        """Test GetScatterer."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)
        for i in range(c.GetNbScatterer()):
            c.GetScatterer(i)
        return

    def testDummyAtom(self):
        """Test dummy atoms."""
        c = makeCrystal(*makeScatterer())

        c.AddScatterer(Atom(0, 0, 0, "dummy", None))

        d = c.GetScatterer("dummy")
        self.assertTrue(d.GetScatteringPower() is None)
        return

    def testEmbedding(self):
        """Test integrity of mutually-embedded objects."""

        c = makeCrystal(*makeScatterer())

        class Level1(object):
            def __init__(self, c):
                self.c = c
                self.level2 = Level2(self)
                return

        class Level2(object):
            def __init__(self, level1):
                self.level1 = level1
                return

        l1 = Level1(c)

        del l1

        return


if __name__ == "__main__":
    unittest.main()
