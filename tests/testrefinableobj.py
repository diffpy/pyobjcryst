#!/usr/bin/env python
"""Tests for refinableobj module."""

from pyobjcryst import *
import unittest

from utils import makeScatterer, makeCrystal

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

    def TestRemoveChild(self):
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



if __name__ == "__main__":

    unittest.main()

