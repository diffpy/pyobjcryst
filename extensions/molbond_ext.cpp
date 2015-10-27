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
* boost::python bindings to ObjCryst::MolBond.
*
* Changes from ObjCryst::MolBond
* - Added __getitem__ access for MolAtoms.
* - File IO is disabled
* - GetDeriv and CalcGradient are not wrapped.
* - Length0, LengthDelta, LengthSigma and BondOrder are wrapped as properties
*   rather than methods.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <ObjCryst/ObjCryst/Molecule.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

MolAtom* _GetAtom(MolBond& mb, size_t i)
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
        default:
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            throw_error_already_set();
    }

    // Do this to avoid ugly compiler warnings
    return rv;
}

} // namespace


void wrap_molbond()
{

    class_<MolBond, bases<Restraint> > ("MolBond", no_init)
        //init<MolAtom&, MolAtom&, const double, const double, const double,
        //Molecule&, const double>())
        .def("GetMolecule", (Molecule& (MolBond::*)())
            &MolBond::GetMolecule,
            return_internal_reference<>())
        .def("GetLogLikelihood",
            (double (MolBond::*)() const)
            &MolBond::GetLogLikelihood)
        .def("GetLogLikelihood",
            (double (MolBond::*)(const bool, const bool) const)
            &MolBond::GetLogLikelihood)
        .def("GetName", &MolBond::GetName)
        .def("GetAtom1", (MolAtom& (MolBond::*)()) &MolBond::GetAtom1,
            return_internal_reference<>())
        .def("GetAtom2", (MolAtom& (MolBond::*)()) &MolBond::GetAtom2,
            return_internal_reference<>())
        .def("SetAtom1", &MolBond::SetAtom1,
            with_custodian_and_ward<1,2>())
        .def("SetAtom2", &MolBond::SetAtom2,
            with_custodian_and_ward<1,2>())
        .def("GetLength", &MolBond::GetLength)
        .def("GetLength0", &MolBond::GetLength0)
        .def("GetLengthDelta", &MolBond::GetLengthDelta)
        .def("GetLengthSigma", &MolBond::GetLengthSigma)
        .def("GetBondOrder", &MolBond::GetBondOrder)
        .def("SetLength0", &MolBond::SetLength0)
        .def("SetLengthDelta", &MolBond::SetLengthDelta)
        .def("SetLengthSigma", &MolBond::SetLengthSigma)
        .def("SetBondOrder", &MolBond::SetBondOrder)
        .def("IsFreeTorsion", &MolBond::IsFreeTorsion)
        .def("SetFreeTorsion", &MolBond::SetFreeTorsion)
        // Python-only
        .add_property("Length", &MolBond::GetLength)
        .add_property("Length0", &MolBond::GetLength0, &MolBond::SetLength0)
        .add_property("LengthDelta", &MolBond::GetLengthDelta,
            &MolBond::SetLengthDelta)
        .add_property("LengthSigma", &MolBond::GetLengthSigma,
            &MolBond::SetLengthSigma)
        .add_property("BondOrder", &MolBond::GetBondOrder,
            &MolBond::SetBondOrder)
        .def("__getitem__", &_GetAtom,
            return_internal_reference<>())
        ;
}
