#!/usr/bin/env python
##############################################################################
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Tests for indexing module."""

import unittest

from numpy import pi

from pyobjcryst.indexing import (
    CellExplorer,
    CrystalCentering,
    CrystalSystem,
    EstimateCellVolume,
    PeakList,
    RecUnitCell,
    quick_index,
)


class TestIndexing(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_estimate_cell_volume(self):
        """Check EstimateCellVolume."""
        # 20 reflections observed from d=47.326A to 1.537A
        v = EstimateCellVolume(
            1 / 1.537,
            1 / 47.326,
            20,
            CrystalSystem.CUBIC,
            CrystalCentering.LATTICE_P,
            1.2,
        )
        self.assertAlmostEqual(v, 309, delta=2)
        v = EstimateCellVolume(
            1 / 1.537,
            1 / 47.326,
            20,
            CrystalSystem.CUBIC,
            CrystalCentering.LATTICE_P,
            0.3,
        )
        self.assertAlmostEqual(v, 2475, delta=2)
        v = EstimateCellVolume(
            1 / 1.537,
            1 / 47.326,
            20,
            CrystalSystem.ORTHOROMBIC,
            CrystalCentering.LATTICE_F,
            1.2,
        )
        self.assertAlmostEqual(v, 308, delta=2)
        v = EstimateCellVolume(
            1 / 1.537,
            1 / 47.326,
            20,
            CrystalSystem.ORTHOROMBIC,
            CrystalCentering.LATTICE_I,
            0.3,
        )
        self.assertAlmostEqual(v, 666, delta=2)

    def test_recunitcell(self):
        r = RecUnitCell(
            0,
            0.1,
            0,
            0,
            0,
            0,
            0,
            CrystalSystem.CUBIC,
            CrystalCentering.LATTICE_P,
            0,
        )
        d = r.hkl2d(1, 1, 1, None, 0)
        self.assertAlmostEqual(d, 0.03, 5)
        u = r.DirectUnitCell()
        self.assertAlmostEqual(u[0], 10, 5)
        self.assertAlmostEqual(u[3], pi / 2)

    def test_quick_index(self):
        # Try to index cimetidine powder pattern from experimental list of points
        v = [
            0.106317,
            0.113542,
            0.146200,
            0.152765,
            0.161769,
            0.166021,
            0.186157,
            0.188394,
            0.189835,
            0.200636,
            0.207603,
            0.211856,
            0.212616,
            0.215067,
            0.220722,
            0.221532,
            0.223939,
            0.227054,
            0.231044,
            0.235053,
        ]
        pl = PeakList()
        pl.set_dobs_list(v)
        ex = quick_index(pl, verbose=False, continue_on_sol=False)
        self.assertGreater(ex.GetBestScore(), 120)
        sols = ex.GetSolutions()
        # Without continue_on_sol=True, this should yield only one solution
        self.assertEqual(len(sols), 1)
        self.assertGreater(sols[0][1], 120)
        ruc = sols[0][0]
        # Check lattice type
        self.assertEqual(ruc.centering, CrystalCentering.LATTICE_P)
        self.assertEqual(ruc.lattice, CrystalSystem.MONOCLINIC)
        # Cell volume
        self.assertAlmostEqual(ruc.DirectUnitCell()[-1], 1280, delta=2)


if __name__ == "__main__":
    unittest.main()
