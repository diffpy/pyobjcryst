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
* boost::python bindings to ObjCryst::MolBondAngle.  
* 
* Changes from ObjCryst++
* - Added __getitem__ access for MolAtoms.
* - File IO is disabled
* - GetDeriv and CalcGradient are not wrapped.
* - Angle0, AngleDelta and AngleSigma are wrapped as properties rather than
*   methods.
* - IsFlexible and SetFlexible are not wrapped, as they are not implemented in
*   the library.
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

MolAtom& _GetAtom(MolBondAngle& mb, size_t i)
{
    switch(i)
    {
        case 0:
            return mb.GetAtom1();
            break;
        case 1:
            return mb.GetAtom2();
            break;
        case 2:
            return mb.GetAtom3();
            break;
        default:
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            throw_error_already_set();
    }
}


} // namespace


BOOST_PYTHON_MODULE(_molbondangle)
{

    class_<MolBondAngle, bases<Restraint> > ("MolBondAngle", no_init)
        .def("GetMolecule", (Molecule& (MolBondAngle::*)()) 
            &MolBondAngle::GetMolecule, 
            return_internal_reference<>())
        .def("GetName", &MolBondAngle::GetName)
        .def("GetLogLikelihood", 
            (float (MolBondAngle::*)() const) 
            &MolBondAngle::GetLogLikelihood) 
        .def("GetLogLikelihood", 
            (float (MolBondAngle::*)(const bool, const bool) const) 
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
        .add_property("angle", &MolBondAngle::GetAngle)
        .add_property("angle0", &MolBondAngle::GetAngle0,
            &MolBondAngle::SetAngle0)
        .add_property("delta", &MolBondAngle::GetAngleDelta,
            &MolBondAngle::SetAngleDelta)
        .add_property("sigma", &MolBondAngle::GetAngleSigma,
            &MolBondAngle::SetAngleSigma)
        .def("__getitem__", &_GetAtom,
            return_internal_reference<>())
        ;
}
