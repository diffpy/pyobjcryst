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
        self.assertTrue(1, len(scl))
        return

    def testGetScatterer(self):
        """Test GetScatterer."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)
        for i in range(c.GetNbScatterer()):
            c.GetScatterer(i)
        return



if __name__ == "__main__":
    unittest.main()


