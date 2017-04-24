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
* boost::python bindings to ObjCryst::MolAtom.
*
* Changes from ObjCryst::MolAtom
* - Wrapped as a to-python converter only.
* - File IO is disabled
* - X, Y and Z are wrapped as properties rather than methods.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Helper function - avoid dereferencing NULL when MolAtom is dummy.

const ScatteringPower* getscatteringpowerpointer(const MolAtom& a)
{
    const ScatteringPower* rv =
        a.IsDummy() ? NULL : &(a.GetScatteringPower());
    return rv;
}


std::string __str__(MolAtom& a)
{
    std::stringstream s;
    s << a.GetName() << " " << a.GetX() << " " << a.GetY() << " " << a.GetZ();
    return s.str();
}

} // namespace


void wrap_molatom()
{

    class_<MolAtom> ("MolAtom", init<const MolAtom&>())
        .def("GetName", (const std::string& (MolAtom::*)() const)
            &MolAtom::GetName,
            return_value_policy<copy_const_reference>())
        .def("SetName", &MolAtom::SetName)
        .def("GetMolecule", (Molecule& (MolAtom::*)())
            &MolAtom::GetMolecule,
            return_internal_reference<>())
        .def("GetX", &MolAtom::GetX)
        .def("GetY", &MolAtom::GetY)
        .def("GetZ", &MolAtom::GetZ)
        .def("GetOccupancy", &MolAtom::GetOccupancy)
        .def("SetX", &MolAtom::SetX)
        .def("SetY", &MolAtom::SetY)
        .def("SetZ", &MolAtom::SetZ)
        .def("SetOccupancy", &MolAtom::SetOccupancy)
        .def("IsDummy", &MolAtom::IsDummy)
        // FIXME - this should be returned as a constant reference. However, I
        // can't get this to work. This returns it as an internal reference,
        // which is probably a bad idea.
        .def("GetScatteringPower", getscatteringpowerpointer,
            return_internal_reference<>())
            //return_value_policy<copy_const_reference>())
        .def("SetIsInRing", &MolAtom::SetIsInRing)
        .def("IsInRing", &MolAtom::IsInRing)
        // Python-only
        .add_property("X", &MolAtom::GetX, &MolAtom::SetX)
        .add_property("Y", &MolAtom::GetY, &MolAtom::SetY)
        .add_property("Z", &MolAtom::GetZ, &MolAtom::SetZ)
        .add_property("Occupancy", &MolAtom::GetOccupancy,
                &MolAtom::SetOccupancy)
        .def("__str__", &__str__)
        ;
}
