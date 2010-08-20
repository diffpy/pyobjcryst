#!/usr/bin/env python
"""Tests for crystal module."""

import unittest

from utils import *

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
        c = makeCrystal(*makeScatterer())
        sp2 = getScatterer()
        self.assertEqual("Ni", sp2.GetName())
        sp3 = sp2
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
        self.assertEquals(1, len(scl))

        sclcopy = scl[:]
        self.assertEquals(1, len(scl))

        del sclcopy[0]
        self.assertEquals(0, len(sclcopy))
        self.assertEquals(1, len(scl))

        del scl[0]
        self.assertEquals(0, len(scl))

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

    def testPickling(self):
        """Test pickling of a crystal."""
        c = makeCrystal(*makeScatterer())

        import pickle
        p = pickle.dumps(c)
        c2 = pickle.loads(p)

        self.assertAlmostEquals(c.a, c2.a, 15)
        self.assertAlmostEquals(c.b, c2.b, 15)
        self.assertAlmostEquals(c.c, c2.c, 15)
        self.assertAlmostEquals(c.alpha, c2.alpha, 15)
        self.assertAlmostEquals(c.beta, c2.beta, 15)
        self.assertAlmostEquals(c.gamma, c2.gamma, 15)

        self.assertEquals(c.GetNbScatterer(), c2.GetNbScatterer())

        s = c.GetScatterer("Ni")
        s2 = c2.GetScatterer("Ni")
        self.assertAlmostEquals(s.X, s2.X, 15)
        self.assertAlmostEquals(s.Y, s2.Y, 15)
        self.assertAlmostEquals(s.Z, s2.Z, 15)

        sp = c.GetScatteringPower("Ni")
        sp2 = c2.GetScatteringPower("Ni")
        self.assertAlmostEquals(sp.Biso, sp2.Biso, 15)
        return


if __name__ == "__main__":
    unittest.main()

