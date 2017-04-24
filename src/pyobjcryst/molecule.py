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

"""Python wrapping of Molecule.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Molecule

- The public data are not wrapped.
- Added __getitem__ access for MolAtoms.
- AddAtom returns the added MolAtom
- AddBond returns the added MolBond
- AddBondAngle returns the added MolBondAngle
- AddDihedralAngle returns the added MolDihedralAngle
- RemoveAtom returns None, has an indexed version
- RemoveBond returns None, has an indexed version
- RemoveBondAngle returns None, has an indexed version
- RemoveDihedralAngle returns None, has an indexed version
- RemoveRigidGroup returns None
- Added GetNbAtoms
- Added GetNbBonds
- Added GetNbBondAngles
- Added GetNbDihedralAngles
- Added GetNbRigidGroups
- Added GetBond
- Added GetBondAngle
- Added GetDihedralAngle
- Added GetRigidGroup
- FindBond returns the bond if found, None otherwise
- FindBondAngle returns the bond angle if found, None otherwise
- FindDihedralAngle returns the dihedral angle if found, None otherwise
- FindAtom is identical to GetAtom.
- FlipAtomGroup is not wrapped.
- FlipGroup, RotorGroup and StretchModeGroup are not wrapped.
- StretchMode getters are not wrapped
- Quaternion ordinates Q0, Q1, Q2 and Q3 wrapped as properties.

Changes from ObjCryst::MolAtom

- Wrapped as a to-python converter only (no constructor)
- File IO is disabled
- X, Y and Z are wrapped as properties rather than methods.

Changes from ObjCryst::MolBondAngle

- Wrapped as a to-python converter only (no constructor)
- Added __getitem__ access for MolAtoms.
- File IO is disabled
- GetDeriv and CalcGradient are not wrapped.
- Angle0, AngleDelta and AngleSigma are wrapped as properties rather than
  methods.
- IsFlexible and SetFlexible are not wrapped, as they are not implemented in
  the library.

Changes from ObjCryst::MolDihedralAngle

- Wrapped as a to-python converter only (no constructor)
- Added __getitem__ access for MolAtoms.

Changes from ObjCryst::Quaternion

- IO is not wrapped
- Q0, Q1, Q2 and Q3 are wrapped as properties, rather than functions.
- RotateVector overloaded to return tuple of the mutated arguments.

Changes from ObjCryst::RigidGroup

- RigidGroup is wrapped to have python-set methods rather than stl::set
  methods.
"""

__all__ = ["Molecule", "GetBondLength", "GetBondAngle",
           "GetDihedralAngle", "MolAtom", "MolBond",
           "MolBondAngle", "MolDihedralAngle", "Quaternion",
           "RigidGroup", "StretchMode", "StretchModeBondLength",
           "StretchModeBondAngle", "StretchModeTorsion", "StretchModeTwist"]

# TODO - MolRing

from pyobjcryst._pyobjcryst import Molecule
from pyobjcryst._pyobjcryst import GetBondLength
from pyobjcryst._pyobjcryst import GetBondAngle
from pyobjcryst._pyobjcryst import GetDihedralAngle
from pyobjcryst._pyobjcryst import MolAtom
from pyobjcryst._pyobjcryst import MolBond
from pyobjcryst._pyobjcryst import MolBondAngle
from pyobjcryst._pyobjcryst import MolDihedralAngle
from pyobjcryst._pyobjcryst import Quaternion
from pyobjcryst._pyobjcryst import RigidGroup
from pyobjcryst._pyobjcryst import StretchMode
from pyobjcryst._pyobjcryst import StretchModeBondLength
from pyobjcryst._pyobjcryst import StretchModeBondAngle
from pyobjcryst._pyobjcryst import StretchModeTorsion
from pyobjcryst._pyobjcryst import StretchModeTwist
