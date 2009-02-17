/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
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
* boost::python bindings to ObjCryst::MolBond.  
* 
* Changes from ObjCryst++
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Molecule.h"
#include "RefinableObj/RefinableObj.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <string>
#include <sstream>
#include <map>
#include <set>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

} // namespace


BOOST_PYTHON_MODULE(_molbond)
{

    class_<MolBond, bases<Restraint> > ("MolBond", no_init)
        //init<MolAtom&, MolAtom&, const float, const float, const float, 
        //Molecule&, const float>())
        .def("GetName", &MolBond::GetName)
        .def("GetAtom1", (MolAtom& (MolBond::*)()) &MolBond::GetAtom1,
            return_internal_reference<>())
        .def("GetAtom2", (MolAtom& (MolBond::*)()) &MolBond::GetAtom2,
            return_internal_reference<>())
        ;
}
