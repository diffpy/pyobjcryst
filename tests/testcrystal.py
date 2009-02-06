#!/usr/bin/env python
"""Small tests for pyobjcryst.

To check for memory leaks, run
valgrind --tool=memcheck --leak-check=full /usr/bin/python ./pyobjcrysttest.py

"""

from pyobjcryst import *
from numpy import pi
import unittest

def makeScatterer():
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8*pi*pi*0.003)
    atom = Atom(0, 0, 0, "Ni", sp)
    return sp, atom

def makeCrystal(sp, atom):
    c = Crystal(3.52, 3.52, 3.52, "225")
    c.AddScatterer(atom)
    c.AddScatteringPower(sp)
    return c

def getScatterer():
    """Make a crystal and return scatterer from GetScatt."""
    sp, atom = makeScatterer()
    c = makeCrystal(sp, atom)

    sp2 = c.GetScatt(sp.GetName())
    return sp2

class TestObjCryst(unittest.TestCase):

    def testCrystalScope(self):
        """Test to see if the the crystal survives after it is out of scope."""
        sp, atom = makeScatterer()
        makeCrystal(sp, atom)
        # The crystal is out of scope. Since the lifetime of the atom and
        # scatterer are linked, the crystal should stay alive in memory.
        self.assertEqual("Ni", sp.GetName())
        self.assertEqual("Ni", atom.GetName())
    
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


if __name__ == "__main__":
    unittest.main()


