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
* boost::python bindings to ObjCryst::MolBondAngle.
*
* Changes from ObjCryst::MolBondAngle
* - Wrapped as a to-python converter only (no constructor)
* - Added __getitem__ access for MolAtoms.
* - File IO is disabled
* - GetDeriv and CalcGradient are not wrapped.
* - Angle0, AngleDelta and AngleSigma are wrapped as properties rather than
*   methods.
* - IsFlexible and SetFlexible are not wrapped, as they are not implemented in
*   the library.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <ObjCryst/ObjCryst/Molecule.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

MolAtom* _GetAtom(MolBondAngle& mb, size_t i)
{
    MolAtom* rv = NULL;
    switch(i)
    {
        case 0:
            rv = &(mb.GetAtom1());
            break;
        case 1:
            rv = &(mb.GetAtom2());
            break;
        case 2:
            rv = &(mb.GetAtom3());
            break;
        default:
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            throw_error_already_set();
    }
    return rv;
}


} // namespace


void wrap_molbondangle()
{

    class_<MolBondAngle, bases<Restraint> > ("MolBondAngle", no_init)
        .def("GetMolecule", (Molecule& (MolBondAngle::*)())
            &MolBondAngle::GetMolecule,
            return_internal_reference<>())
        .def("GetName", &MolBondAngle::GetName)
        .def("GetLogLikelihood",
            (double (MolBondAngle::*)() const)
            &MolBondAngle::GetLogLikelihood)
        .def("GetLogLikelihood",
            (double (MolBondAngle::*)(const bool, const bool) const)
            &MolBondAngle::GetLogLikelihood)
        .def("GetAngle", &MolBondAngle::GetAngle)
        .def("GetAngle0", &MolBondAngle::GetAngle0)
        .def("GetAngleDelta", &MolBondAngle::GetAngleDelta)
        .def("GetAngleSigma", &MolBondAngle::GetAngleSigma)
        .def("SetAngle0", &MolBondAngle::SetAngle0)
        .def("SetAngleDelta", &MolBondAngle::SetAngleDelta)
        .def("SetAngleSigma", &MolBondAngle::SetAngleSigma)
        .def("GetAtom1", (MolAtom& (MolBondAngle::*)()) &MolBondAngle::GetAtom1,
            return_internal_reference<>())
        .def("GetAtom2", (MolAtom& (MolBondAngle::*)()) &MolBondAngle::GetAtom2,
            return_internal_reference<>())
        .def("GetAtom3", (MolAtom& (MolBondAngle::*)()) &MolBondAngle::GetAtom3,
            return_internal_reference<>())
        .def("SetAtom1", &MolBondAngle::SetAtom1,
            with_custodian_and_ward<1,2>())
        .def("SetAtom2", &MolBondAngle::SetAtom2,
            with_custodian_and_ward<1,2>())
        .def("SetAtom3", &MolBondAngle::SetAtom3,
            with_custodian_and_ward<1,2>())
        //.def("IsFlexible", &MolBondAngle::IsFlexible)
        //.def("SetFlexible", &MolBondAngle::SetFlexible)
        // Python-only
        .add_property("Angle", &MolBondAngle::GetAngle)
        .add_property("Angle0", &MolBondAngle::GetAngle0,
            &MolBondAngle::SetAngle0)
        .add_property("AngleDelta", &MolBondAngle::GetAngleDelta,
            &MolBondAngle::SetAngleDelta)
        .add_property("AngleSigma", &MolBondAngle::GetAngleSigma,
            &MolBondAngle::SetAngleSigma)
        .def("__getitem__", &_GetAtom,
            return_internal_reference<>())
        ;
}
