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

"""Python wrapping of RefinableObj.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::RefinableObj

- XMLOutput and XMLInput accept python file-like objects.
- GetPar that takes a const double* is not exposed, as it is designed for
  internal use.
- GetParamSet returns a copy of the internal data so that no indirect
  manipulation can take place from python.
- SetDeleteRefParInDestructor(false) is called in the constructors of the
  python class and the parameter accessors.
- SetDeleteRefParInDestructor is not exposed.
- RemovePar is overloaded to return None.

Changes from ObjCryst::RefinablePar

- The constructor has been changed to accept a double, rather than a pointer to
  a double.
- The copy and default constructors and Init are not wrapped in order to avoid
  memory corruption. Since boost cannot implicitly handle double* object, a
  wrapper class had to be created. However, this wrapper class cannot be used
  to convert RefinablePar objected created in c++.  Thus,
  ObjCryst::RefinablePar objects created in c++ are passed into python as
  instances of _RefinablePar, which is a python wrapper around
  ObjCryst::RefinablePar. The RefinablePar python class is a wrapper around
  the C++ class PyRefinablePar, which manages its own double*.  These python
  classes are interchangable once instantiated, so users should not notice.
- XML input/output are not exposed.

Changes from ObjCryst::RefinableObjClock

- operator= is wrapped as the SetEqual method
  a.SetEqual(b) -> a = b

Changes from ObjCryst::ObjRegistry

- DeleteAll not wrapped
- GetObj(const unsigned int i) not wrapped. Documentation says that this is
  for internal use only.

Changes from ObjCryst::Restraint

- The default and copy constructors are not wrapped, nor is Init.
- GetType returns a non-const reference to the RefParType.  This should be a
  no-no, but RefParType has no mutating methods, so this should no lead to
  trouble.
- XML input/output are not exposed.
"""

__all__ = ["RefinableObjClock", "RefinableObj", "RefObjOpt",
           "RefinableObjRegistry", "RefParType", "RefParDerivStepModel",
           "RefinablePar", "Restraint", "ScattererRegistry",
           "ScatteringPowerRegistry", "ZAtomRegistry"]

from pyobjcryst._pyobjcryst import RefinableObjClock
from pyobjcryst._pyobjcryst import RefinableObj
from pyobjcryst._pyobjcryst import RefObjOpt
from pyobjcryst._pyobjcryst import RefinableObjRegistry
from pyobjcryst._pyobjcryst import RefParType
from pyobjcryst._pyobjcryst import RefParDerivStepModel
from pyobjcryst._pyobjcryst import RefinablePar
from pyobjcryst._pyobjcryst import Restraint
from pyobjcryst._pyobjcryst import ScattererRegistry
from pyobjcryst._pyobjcryst import ScatteringPowerRegistry
from pyobjcryst._pyobjcryst import ZAtomRegistry
