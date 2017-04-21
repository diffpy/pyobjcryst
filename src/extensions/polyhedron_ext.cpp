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
* boost::python bindings to ObjCryst::Polyhedron module.
*
*****************************************************************************/

#include <boost/python/def.hpp>
#include <boost/python/with_custodian_and_ward.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/manage_new_object.hpp>

#include <ObjCryst/ObjCryst/Polyhedron.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

void wrap_polyhedron()
{

    def("MakeTetrahedron", &MakeTetrahedron,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakeOctahedron", &MakeOctahedron,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakeSquarePlane", &MakeSquarePlane,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakeCube", &MakeCube,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakeAntiPrismTetragonal", &MakeAntiPrismTetragonal,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakePrismTrigonal", &MakePrismTrigonal,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakeIcosahedron", &MakeIcosahedron,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

    def("MakeTriangle", &MakeTriangle,
        (bp::arg("cryst"), bp::arg("name"), bp::arg("centralAtom"),
         bp::arg("peripheralAtom"), bp::arg("dist")),
        with_custodian_and_ward_postcall<0,3,
            with_custodian_and_ward_postcall<0,4,
                return_value_policy<manage_new_object> > >());

}
