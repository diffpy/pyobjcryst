#!/usr/bin/env python
"""Functional tests for the RefinableObj/ScatteringPower method surface
restored on Scatterer/ScatteringPowerAtom (see
nanobind_migration_notes.md's "Follow-up: restoring RefinableObj's method
surface on virtually-inherited classes"; presence-only coverage across every
affected class lives in test_virtualbase_forwarding.py). These exercise the
actual behaviour, not just that the method exists.
"""

import io
import unittest

from testutils import makeCrystal, makeScatterer

from pyobjcryst import ObjCrystException
from pyobjcryst.refinableobj import RefinableObjClock


class TestScattererForwarding(unittest.TestCase):
    """Atom reaches RefinableObj only through Scatterer's C++ virtual base."""

    def setUp(self):
        self.sp, self.atom = makeScatterer()
        self.crystal = makeCrystal(self.sp, self.atom)

    def test_names(self):
        self.assertEqual("Atom", self.atom.GetClassName())
        self.assertEqual("Ni", self.atom.GetName())
        self.atom.SetName("renamed")
        self.assertEqual("renamed", self.atom.GetName())

    def test_fix_unfix_par(self):
        self.atom.FixAllPar()
        for i in range(self.atom.GetNbPar()):
            self.assertTrue(self.atom.GetPar(i).IsFixed())
        self.atom.UnFixAllPar()
        for i in range(self.atom.GetNbPar()):
            self.assertFalse(self.atom.GetPar(i).IsFixed())

    def test_xml_output_input_roundtrip(self):
        self.atom.SetName("original_name")
        buf = io.StringIO()
        self.atom.XMLOutput(buf)
        xml_state = buf.getvalue()
        self.assertIn("<Atom", xml_state)
        self.atom.SetName("changed")
        self.atom.XMLInput(xml_state)
        self.assertEqual("original_name", self.atom.GetName())

    def test_begin_end_optimization(self):
        # Bookkeeping calls forwarded straight through to RefinableObj --
        # just needs to not raise AttributeError/TypeError.
        self.atom.BeginOptimization()
        self.atom.EndOptimization()

    def test_get_log_likelihood_and_restraint_cost(self):
        self.assertEqual(0.0, self.atom.GetLogLikelihood())
        self.assertEqual(0.0, self.atom.GetRestraintCost())

    def test_update_display(self):
        # No-op outside a GUI context, but must be reachable.
        self.atom.UpdateDisplay()

    def test_get_clock_scatt_comp_list(self):
        """Forwards a *protected* Scatterer member (mClockScattCompList);
        unrelated to the virtual-inheritance issue (it's a plain omission)
        but fixed in the same pass -- see PyScatterer in nb_scatterer.cpp."""
        clock = self.atom.GetClockScattCompList()
        self.assertIsInstance(clock, RefinableObjClock)

    def test_get_option_reachable(self):
        # Scatterer itself registers no options, but GetOption/GetNbOption
        # must still be reachable (they weren't, before the fix).
        self.assertEqual(0, self.atom.GetNbOption())

    def test_create_param_set_roundtrip(self):
        p = self.atom.GetPar(0)
        original = p.GetValue()
        save_id = self.atom.CreateParamSet("saved")
        p.SetValue(original + 1)
        self.assertNotAlmostEqual(original, p.GetValue())
        self.atom.RestoreParamSet(save_id)
        self.assertAlmostEqual(original, p.GetValue())
        self.assertEqual("saved", self.atom.GetParamSetName(save_id))


class TestScatteringPowerAtomForwarding(unittest.TestCase):
    """ScatteringPowerAtom inherits ScatteringPower virtually, and
    ScatteringPower inherits RefinableObj virtually too -- a second hop, so
    both bind_refinableobj_forwarding<T>() and
    bind_scatteringpower_forwarding<T>() had to be applied directly to it."""

    def setUp(self):
        self.sp, self.atom = makeScatterer()

    def test_names(self):
        self.assertEqual("ScatteringPowerAtom", self.sp.GetClassName())
        self.assertEqual("Ni", self.sp.GetName())

    def test_print_does_not_raise(self):
        # Print() writes to stdout; just confirm it's reachable and runs.
        self.sp.Print()

    def test_log_likelihood(self):
        self.assertEqual(0.0, self.sp.GetLogLikelihood())

    def test_scatteringpower_own_methods_forwarded(self):
        """These are ScatteringPower's *own* methods (not RefinableObj's),
        forwarded via bind_scatteringpower_forwarding<T>()."""
        self.assertGreaterEqual(self.sp.GetDynPopCorrIndex(), 0)
        self.assertIsNotNone(self.sp.GetLastChangeClock())
        r, g, b = self.sp.GetColourRGB()
        self.assertTrue(all(0.0 <= v <= 1.0 for v in (r, g, b)))

    def test_lsq_methods_reachable_and_raise_when_unset(self):
        """Ported forward alongside the forwarding fix: GetLSQCalc/Obs/
        Weight/Deriv exist on RefinableObj and are therefore forwarded here
        too, even though nothing has configured a custom LSQ target."""
        with self.assertRaises(ObjCrystException):
            self.sp.GetLSQCalc(0)


if __name__ == "__main__":
    unittest.main()
