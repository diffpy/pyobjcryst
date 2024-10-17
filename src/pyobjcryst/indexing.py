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

"""Python wrapping of UnitCell.h

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).
"""

__all__ = ["CrystalSystem", "CrystalCentering", "EstimateCellVolume",
           "RecUnitCell", "PeakList_hkl", "PeakList_hkl0", "PeakList",
           "CellExplorer", "quick_index"]

import time
from numpy import deg2rad
from pyobjcryst._pyobjcryst import CrystalSystem, CrystalCentering, \
    EstimateCellVolume, RecUnitCell, PeakList_hkl, PeakList_hkl0, PeakList,\
    CellExplorer


def quick_index(pl, min_obs_ratio=0.3, max_obs_ratio=1.5, nb_refl=20, try_centered_lattice=True,
                continue_on_sol=False, max_nb_spurious=0, verbose=True):
    if len(pl) > nb_refl:
        pl.resize(nb_refl)
    nb = len(pl)
    dmin = pl.GetPeakList()[nb - 1].dobs
    dmax = pl.GetPeakList()[0].dobs / 10  # assume there are no peaks at lower resolution
    if verbose:
        print("Predicting volumes from %2u peaks between d=%6.3f and d=%6.3f\n" % (nb, 1 / dmax, 1 / dmin))
        print("Starting indexing using %2u peaks" % nb)
    ex = CellExplorer(pl, CrystalSystem.CUBIC, 0)
    ex.SetLengthMinMax(3, 25)
    ex.SetAngleMinMax(deg2rad(90), deg2rad(140))
    ex.SetD2Error(0)
    stop_score = 50
    report_score = 10
    stop_depth = 6 + int(continue_on_sol)
    report_depth = 4
    for nb_spurious in range(0, max_nb_spurious + 1):
        ex.SetNbSpurious(nb_spurious)
        for csys in [CrystalSystem.CUBIC, CrystalSystem.TETRAGONAL, CrystalSystem.RHOMBOEDRAL, CrystalSystem.HEXAGONAL,
                     CrystalSystem.ORTHOROMBIC, CrystalSystem.MONOCLINIC]:
            if csys == CrystalSystem.CUBIC:
                vcen = [CrystalCentering.LATTICE_P, CrystalCentering.LATTICE_I, CrystalCentering.LATTICE_F]
            elif csys == CrystalSystem.TETRAGONAL:
                vcen = [CrystalCentering.LATTICE_P, CrystalCentering.LATTICE_I]
            elif csys == CrystalSystem.RHOMBOEDRAL:
                vcen = [CrystalCentering.LATTICE_P]
            elif csys == CrystalSystem.HEXAGONAL:
                vcen = [CrystalCentering.LATTICE_P]
            elif csys == CrystalSystem.ORTHOROMBIC:
                vcen = [CrystalCentering.LATTICE_P, CrystalCentering.LATTICE_A, CrystalCentering.LATTICE_B,
                        CrystalCentering.LATTICE_C, CrystalCentering.LATTICE_I, CrystalCentering.LATTICE_F]
            elif csys == CrystalSystem.MONOCLINIC:
                vcen = [CrystalCentering.LATTICE_P, CrystalCentering.LATTICE_A,CrystalCentering.LATTICE_C,
                        CrystalCentering.LATTICE_I]

            for cent in vcen:
                centc = 'P'
                if cent == CrystalCentering.LATTICE_I:
                    centc = 'I'
                elif cent == CrystalCentering.LATTICE_A:
                    centc = 'A'
                elif cent == CrystalCentering.LATTICE_B:
                    centc = 'B'
                if cent == CrystalCentering.LATTICE_C:
                    centc = 'C'
                elif cent == CrystalCentering.LATTICE_F:
                    centc = 'F'

                minv = EstimateCellVolume(dmin, dmax, nb, csys, cent, max_obs_ratio)
                maxv = EstimateCellVolume(dmin, dmax, nb, csys, cent, min_obs_ratio)
                ex.SetVolumeMinMax(minv, maxv)
                lengthmax = 3 * maxv ** (1 / 3.)
                if lengthmax < 25:
                    lengthmax = 25
                ex.SetLengthMinMax(3, lengthmax)
                ex.SetCrystalSystem(csys)
                ex.SetCrystalCentering(cent)
                if verbose:
                    print("%11s %c : V= %6.0f -> %6.0f A^3, max length=%6.2fA" %
                          (csys.name, centc, minv, maxv, lengthmax))
                t0 = time.time()
                ex.DicVol(report_score, report_depth, stop_score, stop_depth, verbose=False)
                if verbose:
                    print(" -> %3u sols in %6.2fs, best score=%6.1f\n" %
                          (len(ex.GetSolutions()), time.time() - t0, ex.GetBestScore()))
                if try_centered_lattice:
                    break
    return ex
