##############################################################################
#
# PyObjCryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Utilities for crystals."""

def _xyztostring(crystal):

    nsc = 0
    out = ""
    scl = crystal.GetScatteringComponentList()

    out += "...\n"

    for s in scl:
        sp = s.mpScattPow
        if sp is None:
            continue
        nsc += 1
        el = sp.GetSymbol()
        xyz = [s.X, s.Y, s.Z]
        xyz = crystal.FractionalToOrthonormalCoords(*xyz)
        x, y, z = xyz
        out += "%s %f %f %f\n"%(el, x, y, z)

    out = "%i\n"%nsc + out
    return out


def printxyz(crystal):
    """Print a crystal in xyz format."""

    print _xyztostring(crystal)
    return


def writexyz(crystal, filename):
    """Write a crystal to an xyz file."""

    f = file(filename, 'w')
    out = _xyztostring(crystal)
    f.write(out)
    f.close()
    return
