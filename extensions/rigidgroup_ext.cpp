/*****************************************************************************
*
* PyObjCryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::RigidGroup.  
* 
* Changes from ObjCryst::RigidGroup
* - RigidGroup is wrapped to have python-set methods rather than stl::set
*   methods.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/ObjCryst/Molecule.h"

#include <boost/python.hpp>
#include <boost/python/class.hpp>

#include <set>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

typedef std::set<MolAtom*> MolAtomSet;

namespace {


} // namespace


void wrap_rigidgroup()
{

    class_<RigidGroup, bases<MolAtomSet> >("RigidGroup", init<>())
        .def(init<const RigidGroup&>())
        .def("GetName", &RigidGroup::GetName)
        ;
}
