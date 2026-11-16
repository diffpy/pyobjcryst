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

import gc
import unittest

from testutils import (
    getScatterer,
    makeCrystal,
    makeScatterer,
    makeScattererAnisotropic,
)

from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.scatteringpower import ScatteringPowerAtom


class TestCrystal(unittest.TestCase):

    def testCrystalScope(self):
        """Test to see if the the crystal survives after it is out of
        scope."""
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
        """Make sure we get an error when trying to add or remove
        Null."""
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
        c.AddScatteringPower(sp2)
        c.AddScatterer(atom2)

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
        """Test GetScatterer and GetScatt."""
        sp, atom = makeScatterer()
        c = makeCrystal(sp, atom)
        for fget in (c.GetScatterer, c.GetScatt):
            for i in range(c.GetNbScatterer()):
                fget(i)
            a = fget(0)
            ani = fget("Ni")
            self.assertEqual([a.X, a.Y, a.Z], [ani.X, ani.Y, ani.Z])
            a.X = 0.112
            self.assertEqual(a.X, ani.X)
            aneg = fget(-1)
            self.assertEqual(a.X, aneg.X)
            self.assertRaises(ValueError, fget, "invalid")
            self.assertRaises(IndexError, fget, 10)
            self.assertRaises(IndexError, fget, -2)
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

    def test_display_list(self):
        """Test the creation of a atoms list for display using 3dmol."""
        c = makeCrystal(*makeScatterer())
        c._display_list()
        c._display_list(full_molecule=True)
        c._display_list(enantiomer=True)
        c._display_list(only_independent_atoms=True)

    def testConnectAtomsKeepsHeldAtoms(self):
        """Regression test for #132: ConnectAtoms used to delete atoms
        owned by Python, causing a double free."""
        c = Crystal(10, 10, 10, "P1")
        sp = ScatteringPowerAtom("C", "C")
        c.AddScatteringPower(sp)
        atoms = [
            Atom(0.1 + 0.14 * i, 0.1, 0.1, "C%d" % i, sp) for i in range(4)
        ]
        for atom in atoms:
            c.AddScatterer(atom)
        c.ConnectAtoms()
        self.assertEqual(1, c.GetNbScatterer())
        del c
        gc.collect()
        self.assertEqual(
            ["C0", "C1", "C2", "C3"], [a.GetName() for a in atoms]
        )

    def testDefaultCrystalKeepsPythonAtom(self):
        """Crystal() deletes its sub-objects in its destructor, but not
        those owned by Python."""
        c = Crystal()
        sp = ScatteringPowerAtom("C", "C")
        c.AddScatteringPower(sp)
        atom = Atom(0.1, 0.1, 0.1, "C0", sp)
        c.AddScatterer(atom)
        del c
        gc.collect()
        self.assertEqual("C0", atom.GetName())
        self.assertEqual("C", sp.GetName())

    def testRemoveHeldScatterer(self):
        """A removed scatterer stays valid while Python holds it."""
        c = Crystal(10, 10, 10, "P1")
        sp = ScatteringPowerAtom("C", "C")
        c.AddScatteringPower(sp)
        c.AddScatterer(Atom(0.1, 0.1, 0.1, "C0", sp))
        c.AddScatterer(Atom(0.3, 0.1, 0.1, "C1", sp))
        atom = c.GetScatt("C0")
        c.RemoveScatterer(atom)
        self.assertEqual(1, c.GetNbScatterer())
        self.assertEqual("C1", c.GetScatt(0).GetName())
        del c
        gc.collect()
        self.assertEqual("C0", atom.GetName())
        self.assertAlmostEqual(0.1, atom.GetX())


def test_connect_atoms_keeps_held_atoms(loadcifdata):
    # Test ConnectAtoms on a CIF crystal while Python holds its atoms
    # Expected: Removed atoms stay valid after the crystal is deleted.
    crystal = loadcifdata("paracetamol.cif")
    held = [crystal.GetScatt(i) for i in range(crystal.GetNbScatterer())]
    names = [a.GetName() for a in held]
    crystal.ConnectAtoms()
    assert crystal.GetNbScatterer() < len(held)
    del crystal
    gc.collect()
    assert [a.GetName() for a in held] == names


def test_xml_input_keeps_held_scatterers(loadcifdata):
    # Test XMLInput into a Crystal() while Python holds its scatterers
    # Expected: The old scatterers stay valid and the crystal is replaced.
    from pyobjcryst.refinableobj import RefinableObj

    crystal = Crystal()
    RefinableObj.XMLInput(crystal, loadcifdata("caffeine.cif").xml())
    held = [crystal.GetScatt(i) for i in range(crystal.GetNbScatterer())]
    names = [a.GetName() for a in held]
    paracetamol = loadcifdata("paracetamol.cif")
    RefinableObj.XMLInput(crystal, paracetamol.xml())
    assert crystal.GetNbScatterer() == paracetamol.GetNbScatterer()
    del crystal
    gc.collect()
    assert [a.GetName() for a in held] == names


if __name__ == "__main__":
    unittest.main()
