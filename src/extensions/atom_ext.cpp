/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::Atom.
*
* Changes from ObjCryst::Atom
* - The default constructor has been disabled. When not followed-up by Init, it
*   will cause segmentation faults, even if it is printed.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>

#include <string>

#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Helper function - avoid dereferencing NULL when Atom is dummy.

const ScatteringPower* getscatteringpowerpointer(const Atom& a)
{
    const ScatteringPower* rv =
        a.IsDummy() ? NULL : &(a.GetScatteringPower());
    return rv;
}

}   // namespace

void wrap_atom()
{
    class_<Atom, bases<Scatterer> >("Atom", init<const Atom&>(bp::arg("old")))
        // Constructors
        .def(init<const double, const double, const double, const std::string&,
            const ObjCryst::ScatteringPower*, bp::optional<const double> >(
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
        .def("GetScatteringPower", &getscatteringpowerpointer,
            return_internal_reference<>())
            //return_value_policy<copy_const_reference>())
        ;
}
