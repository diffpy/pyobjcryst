#!/usr/bin/env python
"""Tests for crystal module."""

from pyobjcryst import *
import unittest

from utils import *


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
    
    def testMultiAdd(self):
        """Test exception for multi-crystal additions."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)
        # Force this exception
        self.assertRaises(AttributeError, makeCrystal, sp, atom)
        return

    def testScattererScope(self):
        """Test when atoms go out of scope before crystal."""
        c = makeCrystal(*makeScatterer())
        sp2 = getScatterer()
        self.assertEqual("Ni", sp2.GetName())
        return

    def testRemoveFunctions(self):
        """Test the RemoveScatterer and RemoveScatteringPower method."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)

        sp2, atom2 = makeScatterer()
        self.assertRaises(AttributeError, c.AddScatterer, atom2)
        self.assertRaises(AttributeError, c.AddScatteringPower, sp2)

        # Remove the scatterers
        c.RemoveScatterer(atom)
        c.RemoveScatteringPower(sp)
        # Remove again
        self.assertRaises(AttributeError, c.RemoveScatterer, atom)
        self.assertRaises(AttributeError, c.RemoveScatteringPower, sp)

        # Try to remove scatterers that are not in the crystal
        self.assertRaises(AttributeError, c.RemoveScatterer, atom2)
        self.assertRaises(AttributeError, c.RemoveScatteringPower, sp2)
        return

    def testGetScatteringComponentList(self):
        """Test the RemoveScatterer and RemoveScatteringPower method."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)
        scl = c.GetScatteringComponentList()
        self.assertTrue(1, len(scl))
        return



if __name__ == "__main__":
    unittest.main()


