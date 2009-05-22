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
* boost::python bindings to ObjCryst::Atom.
*
* Changes from ObjCryst::Atom
* - The default constructor has been disabled. When not followed-up by Init, it
*   will cause segmentation faults, even if it is printed.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Atom.h"
#include "ObjCryst/Scatterer.h"
#include "ObjCryst/ScatteringPower.h"
#include "CrystVector/CrystVector.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

}

BOOST_PYTHON_MODULE(_atom)
{

    class_<Atom, bases<Scatterer> >("Atom", init<const Atom&>((bp::arg("old"))))
        // Constructors
        .def(init<const float, const float, const float, const std::string&, 
            const ObjCryst::ScatteringPower*, optional<const float> >(
            (bp::arg("x"), bp::arg("y"), bp::arg("z"), bp::arg("name"),
             bp::arg("pow"), bp::arg("popu")))
            [with_custodian_and_ward<1,6>()])
        // Methods
        .def("Init", &Atom::Init, 
            (bp::arg("x"), bp::arg("y"), bp::arg("z"), bp::arg("name"),
             bp::arg("pow"), bp::arg("popu")=1.0),
            with_custodian_and_ward<1,6>())
        .def("GetMass", &Atom::GetMass)
        .def("GetRadius", &Atom::GetRadius)
        .def("IsDummy", &Atom::IsDummy)
        // FIXME - this should be returned as a constant reference. However, I
        // can't get this to work. This returns it as an internal reference,
        // which is probably a bad idea.
        .def("GetScatteringPower", &Atom::GetScatteringPower,
            return_internal_reference<>())
            //return_value_policy<copy_const_reference>())
        ;

}
