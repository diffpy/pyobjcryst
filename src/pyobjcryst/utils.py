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

"""Utilities for crystals."""


# FIXME: check if this function does any meaningful job.

def putAtomsInMolecule(crystal, alist = None, name = None):
    """Place atoms from a crystal into a molecule inside the crystal.

    Selected atoms are put into a new Molecule object, which is then placed
    inside of the Crystal. The atoms are then removed from the crystal. The
    molecule is placed at the center of mass of the moved atoms.

    crystal --  The crystal containing the atoms.
    alist   --  A list of indices or names identifying the atoms. If alist is
                None (default), all atoms from the crystal are placed into a
                molecule.
    name    --  A name for the molecule. If name is None (default), the name
                m_cname will be given, where cname is substituted for the
                crystal's name.

    Raises TypeError if idxlist identifies a non-atom.

    """
    c = crystal

    if name is None:
        name = "m_%s" % c.GetName()

    if alist is None:
        alist = range(c.GetNbScatterer())

    from pyobjcryst.molecule import Molecule
    from pyobjcryst.atom import Atom
    m = Molecule(c, name)

    # center of mass
    cx = cy = cz = 0.0

    # mapping fractional coords back into [0, 1)
    from math import floor
    f = lambda v: v - floor(v)

    scat = []
    for idx in alist:
        s = c.GetScatt(idx)
        if not isinstance(s, Atom):
            raise TypeError("identifier '%s' does not specify an Atom")
        sp = s.GetScatteringPower()
        scat.append(s)
        x, y, z = map(f, [s.X, s.Y, s.Z])
        x, y, z = c.FractionalToOrthonormalCoords(x, y, z)
        m.AddAtom(x, y, z, sp, s.GetName())

        cx += x
        cy += y
        cz += z

    # Adjust center of mass
    cx /= len(alist)
    cy /= len(alist)
    cz /= len(alist)

    # Remove scatterers from the crystal
    for s in scat:
        c.RemoveScatterer(s)

    # Put the molecule at the CoM
    m.X, m.Y, m.Z = c.OrthonormalToFractionalCoords(cx, cy, cz)

    # Add the molecule to the crystal
    c.AddScatterer(m)

    m.UpdateScattCompList()

    return


def _xyztostring(crystal):
    """Helper function to write xyz coordinates of a crystal to a string."""

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
    print(_xyztostring(crystal))
    return


def writexyz(crystal, filename):
    """Write a crystal to an xyz file."""
    f = open(filename, 'w')
    out = _xyztostring(crystal)
    f.write(out)
    f.close()
    return
