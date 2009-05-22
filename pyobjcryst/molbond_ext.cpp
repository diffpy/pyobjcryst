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
* - File IO is disabled
* - GetDeriv and CalcGradient are not wrapped.
* - Length0, LengthDelta, LengthSigma and BondOrder are wrapped as properties
*   rather than methods.
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
        .def("GetMolecule", (Molecule& (MolBond::*)()) 
            &MolBond::GetMolecule, 
            return_internal_reference<>())
        .def("GetLogLikelihood", 
            (float (MolBond::*)() const) 
            &MolBond::GetLogLikelihood) 
        .def("GetLogLikelihood", 
            (float (MolBond::*)(const bool, const bool) const) 
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
        .add_property("length", &MolBond::GetLength)
        .add_property("length0", &MolBond::GetLength0, &MolBond::SetLength0)
        .add_property("delta", &MolBond::GetLengthDelta,
            &MolBond::SetLengthDelta)
        .add_property("sigma", &MolBond::GetLengthSigma,
            &MolBond::SetLengthSigma)
        .add_property("order", &MolBond::GetBondOrder,
            &MolBond::SetBondOrder)
        ;
}
