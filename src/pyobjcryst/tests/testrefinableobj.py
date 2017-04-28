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

"""Tests for refinableobj module."""

from pyobjcryst.refinableobj import RefinableObjClock, RefParType, Restraint
from pyobjcryst.refinableobj import RefinablePar, RefinableObj
from pyobjcryst import ObjCrystException

import unittest
import numpy

from pyobjcryst.tests.pyobjcrysttestutils import makeScatterer, makeCrystal

class TestRefinableObjClock(unittest.TestCase):

    def testRelations(self):
        """Test clicking!

        Chances are that someone will someday read this code for an example on
        how to use clocks. If not, then I've wasted my time writing this.
        Anyway, clocks are more complex then they appear. This is because
        ObjCryst++ has an internal clock that gets incremented whenever any
        clock is Clicked. So, one cannot trust that a clock will increment by
        only one value when it is clicked. Furthermore, clocks only alert their
        parents to a change. So, it is possible to decrease the value of a
        parent clock with SetEqual below the values of its children clocks.
        Callling Click on the parent or child will restore the proper parent >
        child relationship.
        """
        c1 = RefinableObjClock()
        c2 = RefinableObjClock()
        parent = RefinableObjClock()
        ref = RefinableObjClock()
        ref2 = RefinableObjClock()
        c1.Reset()
        c2.Reset()
        parent.Reset()
        ref.Reset()
        ref2.Reset()

        # See if these things are at the same spot
        self.assertTrue( not ( c1 < c2 or c2 < c1 ) )

        # Click one to make greater than other
        c1.Click()
        self.assertTrue( c1 > c2 )
        self.assertFalse( c2 > c1)

        # Adding children advances the parent beyond all children
        parent.AddChild(c1)
        parent.AddChild(c2)
        self.assertTrue( parent > c1 )
        self.assertTrue( parent > c2 )
        self.assertTrue( c1 > c2 )

        # Clicking parent leaves the children
        ref.SetEqual(c1)
        parent.Click()
        self.assertTrue( parent > ref )
        self.assertTrue( parent > c1 )
        self.assertTrue( parent > c2 )
        self.assertTrue( c1 > c2 )
        self.assertTrue( not (c1 < ref or ref < c1 ) )

        # Resetting parent does not reset children
        parent.Reset()
        self.assertTrue( parent < c1 )

        # Resetting child does not affect parent
        ref.SetEqual(parent)
        c1.Reset()
        self.assertTrue( not (parent < ref or ref < parent) )

        # Clicking children advances parent
        ref.SetEqual(parent)
        c1.Click()
        self.assertTrue( parent > c1 )
        self.assertTrue( parent > c2 )

        # Reset child does not affect parent or other children
        ref.SetEqual(parent)
        ref2.SetEqual(c1)
        c2.Reset()
        self.assertTrue( not (parent < ref or ref < parent) )
        self.assertTrue( not (c1 < ref2 or ref2 < c1 ) )

        # Increasing child above parent with SetEqual will increase parent to
        # child's value
        ref.SetEqual(parent)
        ref2.SetEqual(c1)
        ref.Click()
        ref.Click()
        self.assertTrue(ref > parent)
        c1.SetEqual(ref)
        self.assertTrue(c1 > ref2)
        self.assertTrue(not ( parent < c1 or c1 < parent ) )
        ref.Reset()
        self.assertTrue( parent > ref )

        # Decreasing child with SetEqual will not affect parent.
        c1.Click()
        ref2.SetEqual(c1)
        c1.Click()
        self.assertTrue(ref2 < c1)
        self.assertTrue(ref2 < parent)
        ref.SetEqual(parent)
        c1.SetEqual(ref2)
        self.assertTrue(not (ref < parent or parent < ref) )

        # Increasing child with SetEqual, so that it is still smaller than
        # parent, will increment parent
        parent.SetEqual(ref2)
        parent.Click()
        self.assertTrue(c1 < parent)
        self.assertTrue(ref2 < parent)
        ref.SetEqual(parent)
        c1.SetEqual(ref2)
        self.assertTrue(c1 < parent)
        self.assertTrue(not (ref < parent or parent < ref) )

        # Reducing parent with SetEqual will not affect children.
        ref.Reset()
        c1.Click()
        c2.Click()
        parent.SetEqual(ref)
        self.assertTrue(parent < c1)
        self.assertTrue(parent < c2)
        return

    def testRemoveChild(self):
        """Test the RemoveChild method."""
        c1 = RefinableObjClock()
        c2 = RefinableObjClock()
        parent = RefinableObjClock()

        # Test equality values after removing child
        parent.AddChild(c1)
        c1.Click()
        self.assertTrue(parent > c1)
        parent.RemoveChild(c1)
        c1.Click()
        self.assertTrue(c1 > parent)

        # Try to remove a clock that is not part of another. This should do
        # nothing, just as in ObjCryst++.
        parent.RemoveChild(c2)
        return


class TestRestraint(unittest.TestCase):

    def testEquality(self):
        """See if we get back what we put in."""
        rpt = RefParType("test")
        res1 = Restraint(rpt)
        rpt2 = res1.GetType()
        self.assertEqual(rpt2, rpt)
        return


class TestRefinablePar(unittest.TestCase):

    def setUp(self):
        self.rpt = RefParType("test")
        self.testpar = RefinablePar("test", 3.0, 0, 10, self.rpt)
        return

    def testToFromPython(self):
        """See if refinable parameters can be created from within python and
        within c++."""
        c = makeCrystal(*makeScatterer())

        # Get a parameter created from c++
        par = c.GetPar("a")
        self.assertAlmostEqual(3.52, par.GetValue())

        # pass a parameter and pass it into c++
        c.AddPar(self.testpar);

        # get it back
        testpar2 = c.GetPar("test")

        self.assertAlmostEqual(self.testpar.GetValue(), testpar2.GetValue())

        testpar2.SetValue(2.17)
        self.assertAlmostEqual(2.17, testpar2.GetValue(), places = 6)
        self.assertAlmostEqual(self.testpar.GetValue(), testpar2.GetValue())
        return

    def testGetType(self):
        """See if we can get the proper RefParType from a RefinablePar."""
        rpt2 = self.testpar.GetType()
        self.assertEqual(rpt2, self.rpt)
        return

class TestRefinableObj(unittest.TestCase):

    def setUp(self):
        """Make a RefinableObj and add some RefinablePars."""
        self.r = RefinableObj()
        self.r.SetName("test1")
        # Add some parameters
        self.rpt = RefParType("test")
        p1 = RefinablePar("p1", 3, 0, 10, self.rpt)
        p2 = RefinablePar("p2", -3, -10, 0, self.rpt)
        self.r.AddPar(p1)
        self.r.AddPar(p2)
        return

    def _getPars(self):
        """Convenience function."""
        p1 = self.r.GetPar(0)
        p2 = self.r.GetPar(1)
        return p1, p2

    def testNames(self):
        """Test the naming methods."""
        self.assertEqual("RefinableObj", self.r.GetClassName())
        self.assertEqual("test1", self.r.GetName())
        return

    def testGetPar(self):
        """Test GetPar."""
        p1 = self.r.GetPar(0)
        p2 = self.r.GetPar(1)
        self.assertEqual(2, self.r.GetNbPar())
        self.assertEqual("p1", p1.GetName())
        self.assertEqual("p2", p2.GetName())
        return

    def testFixUnFix(self):
        """Test FixAllPar."""
        p1, p2 = self._getPars()
        r = self.r

        r.FixAllPar()
        self.assertTrue(p1.IsFixed())
        self.assertTrue(p2.IsFixed())
        r.PrepareForRefinement()
        self.assertEqual(0, r.GetNbParNotFixed())

        r.UnFixAllPar()
        self.assertFalse(p1.IsFixed())
        self.assertFalse(p2.IsFixed())
        r.PrepareForRefinement()
        self.assertEqual(2, r.GetNbParNotFixed())

        r.FixAllPar()
        self.assertTrue(p1.IsFixed())
        self.assertTrue(p2.IsFixed())

        r.SetParIsFixed(0, True)
        r.SetParIsFixed(1, False)
        self.assertTrue(p1.IsFixed())
        self.assertFalse(p2.IsFixed())
        r.PrepareForRefinement()
        self.assertEqual(1, r.GetNbParNotFixed())

        r.SetParIsFixed("p1", False)
        r.SetParIsFixed("p2", True)
        self.assertFalse(p1.IsFixed())
        self.assertTrue(p2.IsFixed())
        r.PrepareForRefinement()
        self.assertEqual(1, r.GetNbParNotFixed())
        return

    def testUsedUnUsed(self):
        """Test FixAllPar."""
        p1, p2 = self._getPars()
        r = self.r

        r.SetParIsUsed("p1", False)
        r.SetParIsUsed("p2", True)
        self.assertFalse(p1.IsUsed())
        self.assertTrue(p2.IsUsed())

        r.SetParIsUsed(self.rpt, True)
        self.assertTrue(p1.IsUsed())
        self.assertTrue(p2.IsUsed())
        return

    def testAddParRefinableObj(self):
        """Test adding another object."""
        r2 = RefinableObj()
        r2.SetName("test2")
        # Add some parameters
        p3 = RefinablePar("p3", 3, 0, 10, self.rpt)
        p4 = RefinablePar("p4", -3, -10, 0, self.rpt)
        r2.AddPar(p3)
        r2.AddPar(p4)

        self.r.AddPar(r2)
        self.assertEqual(4, self.r.GetNbPar())
        return

    def testAddParTwice(self):
        """Try to add the same parameter twice.

        We could stop this in the bindings, but since RefinableObj doesn't
        delete its parameters in the destructor, it shouldn't lead to trouble.
        """
        p3 = RefinablePar("p3", 3, 0, 10, self.rpt)
        self.r.AddPar(p3)
        self.r.AddPar(p3)
        return

    def testParmSets(self):
        """Test creation of parameter sets."""
        self.assertRaises(ObjCrystException, self.r.SaveParamSet, 3)
        p1, p2 = self._getPars()
        r = self.r

        # Test saving and retrieval of parameters
        save1 = r.CreateParamSet("save1")
        savevals1 = r.GetParamSet(save1)
        self.assertTrue( numpy.array_equal([3,-3], savevals1) )

        # Change a parameter test new value
        p1.SetValue(8.0)
        save2 = r.CreateParamSet("save2")
        savevals2 = r.GetParamSet(save2)
        self.assertTrue( numpy.array_equal([8,-3], savevals2) )

        # Restore the old set
        r.RestoreParamSet(save1)
        self.assertEqual(3, p1.GetValue())

        # Get the names
        self.assertEqual(r.GetParamSetName(save1), "save1")
        self.assertEqual(r.GetParamSetName(save2), "save2")

        # Delete parameter sets
        r.ClearParamSet(save2)
        self.assertRaises(ObjCrystException, r.SaveParamSet, save2)
        r.EraseAllParamSet()
        self.assertRaises(ObjCrystException, r.SaveParamSet, save1)

        return

    def testLimits(self):
        """Test the limit-setting functions."""
        p1, p2 = self._getPars()
        r = self.r

        # Check setting absolute limits by name
        r.SetLimitsAbsolute("p1", 0, 1)
        p1.SetValue(8)
        self.assertEqual(1, p1.GetValue())
        p1.SetValue(-1)
        self.assertEqual(0, p1.GetValue())

        # Check setting absolute limits by type
        r.SetLimitsAbsolute(self.rpt, 0, 1)
        p1.SetValue(10)
        p2.SetValue(10)
        self.assertEqual(1, p1.GetValue())
        self.assertEqual(1, p2.GetValue())

        # Check setting relative limits by name
        r.SetLimitsRelative("p1", 0, 1)
        p1.SetValue(8)
        self.assertEqual(2, p1.GetValue())
        p1.SetValue(-1)
        self.assertEqual(1, p1.GetValue())

        # Check setting relative limits by type
        r.SetLimitsRelative(self.rpt, 0, 1)
        p1.SetValue(10)
        p2.SetValue(10)
        self.assertEqual(2, p1.GetValue())
        self.assertEqual(2, p2.GetValue())

        # Check setting proportional limits by name
        p1.SetValue(1)
        r.SetLimitsProportional("p1", 0, 3)
        p1.SetValue(8)
        self.assertEqual(3, p1.GetValue())
        p1.SetValue(-1)
        self.assertEqual(0, p1.GetValue())

        # Check setting proportional limits by type
        p1.SetValue(1)
        p2.SetValue(2)
        r.SetLimitsProportional(self.rpt, 1, 2)
        p1.SetValue(10)
        p2.SetValue(10)
        self.assertEqual(2, p1.GetValue())
        self.assertEqual(4, p2.GetValue())
        return

    def testOptimStep(self):
        """Test SetGlobalOptimStep."""
        p1, p2 = self._getPars()
        self.r.SetGlobalOptimStep(self.rpt, 1)
        self.r.SetGlobalOptimStep(self.rpt, 0.1)
        self.assertAlmostEqual(0.1, p1.GetGlobalOptimStep())
        self.assertAlmostEqual(0.1, p2.GetGlobalOptimStep())
        return


if __name__ == "__main__":
    unittest.main()
