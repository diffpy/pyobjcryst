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
* boost::python bindings to ObjCryst::MolDihedralAngle.
*
* Changes from ObjCryst::MolDihedralAngle
* - Wrapped as a to-python converter only (no constructor)
* - Added __getitem__ access for MolAtoms.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <ObjCryst/ObjCryst/Molecule.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

MolAtom* _GetAtom(MolDihedralAngle& mb, size_t i)
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
        case 3:
            rv = &(mb.GetAtom4());
            break;
        default:
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            throw_error_already_set();
    }

    return rv;
}

} // namespace


void wrap_moldihedralangle()
{

    class_<MolDihedralAngle, bases<Restraint> > ("MolDihedralAngle", no_init)
        .def("GetMolecule", (Molecule& (MolDihedralAngle::*)())
            &MolDihedralAngle::GetMolecule,
            return_internal_reference<>())
        .def("GetName", &MolDihedralAngle::GetName)
        .def("GetLogLikelihood",
            (double (MolDihedralAngle::*)() const)
            &MolDihedralAngle::GetLogLikelihood)
        .def("GetLogLikelihood",
            (double (MolDihedralAngle::*)(const bool, const bool) const)
            &MolDihedralAngle::GetLogLikelihood)
        .def("GetAngle", &MolDihedralAngle::GetAngle)
        .def("GetAngle0", &MolDihedralAngle::GetAngle0)
        .def("GetAngleDelta", &MolDihedralAngle::GetAngleDelta)
        .def("GetAngleSigma", &MolDihedralAngle::GetAngleSigma)
        .def("SetAngle0", &MolDihedralAngle::SetAngle0)
        .def("SetAngleDelta", &MolDihedralAngle::SetAngleDelta)
        .def("SetAngleSigma", &MolDihedralAngle::SetAngleSigma)
        .def("GetAtom1", (MolAtom& (MolDihedralAngle::*)())
            &MolDihedralAngle::GetAtom1,
            return_internal_reference<>())
        .def("GetAtom2", (MolAtom& (MolDihedralAngle::*)())
            &MolDihedralAngle::GetAtom2,
            return_internal_reference<>())
        .def("GetAtom3", (MolAtom& (MolDihedralAngle::*)())
            &MolDihedralAngle::GetAtom3,
            return_internal_reference<>())
        .def("GetAtom4", (MolAtom& (MolDihedralAngle::*)())
            &MolDihedralAngle::GetAtom4,
            return_internal_reference<>())
        .def("SetAtom1", &MolDihedralAngle::SetAtom1,
            with_custodian_and_ward<1,2>())
        .def("SetAtom2", &MolDihedralAngle::SetAtom2,
            with_custodian_and_ward<1,2>())
        .def("SetAtom3", &MolDihedralAngle::SetAtom3,
            with_custodian_and_ward<1,2>())
        .def("SetAtom4", &MolDihedralAngle::SetAtom4,
            with_custodian_and_ward<1,2>())
        // Python-only
        .add_property("Angle", &MolDihedralAngle::GetAngle)
        .add_property("Angle0", &MolDihedralAngle::GetAngle0,
            &MolDihedralAngle::SetAngle0)
        .add_property("AngleDelta", &MolDihedralAngle::GetAngleDelta,
            &MolDihedralAngle::SetAngleDelta)
        .add_property("AngleSigma", &MolDihedralAngle::GetAngleSigma,
            &MolDihedralAngle::SetAngleSigma)
        .def("__getitem__", &_GetAtom,
            return_internal_reference<>())
        ;
}
