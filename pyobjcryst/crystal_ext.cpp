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
* boost::python bindings to ObjCryst::Crystal.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Crystal.h"
#include "ObjCryst/UnitCell.h"
#include "ObjCryst/Atom.h"
#include "CrystVector/CrystVector.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

#include <string>
#include <iostream>

using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overloaded for object management

class CrystalWrap : public Crystal, 
                      public wrapper<Crystal>
{

    public:

    CrystalWrap() : Crystal() {}
    CrystalWrap(const float a, const float b, const float c, const std::string& sgid) :
        Crystal(a, b, c, sgid)  {}
    CrystalWrap(const float a, const float b, const float c, 
            const float alpha, const float beta, const float gamma,
            const std::string& sgid) :
        Crystal(a, b, c, alpha, beta, gamma, sgid)  {}
    CrystalWrap(const CrystalWrap& C) : Crystal(C) {}

    // Crystal::RemoveScatterer normally deletes the memory for the scattering
    // object, and doesn't care if it leaves dangling references.
    void RemoveScatterer(Scatterer *scatt)
    {
        ObjRegistry<Scatterer>& mSR = GetScattererRegistry();
        mSR.DeRegister(*scatt);
        scatt->DeRegisterClient(*this);
        this->RemoveSubRefObj(*scatt);
        // FIXME - There is no way to click the clock!
    }


}; // CrystalWrap

}


BOOST_PYTHON_MODULE(_crystal)
{

    
    class_<CrystalWrap, boost::noncopyable, bases<UnitCell> > ("Crystal", init<>())
        // Constructors
        .def(init<const float, const float, const float, const std::string&>())
        .def(init<const float, const float, const float, 
            const float, const float, const float, 
            const std::string&>())
        .def(init<const CrystalWrap&>())
        // Methods
        // I don't know why this works, but it does.
        .def("AddScatterer", &Crystal::AddScatterer,
                with_custodian_and_ward<2,1, with_custodian_and_ward<1,2> >())
        // FIXME This doesn't update the clock!
        .def("RemoveScatterer", &CrystalWrap::RemoveScatterer)
        .def("GetNbScatterer", &Crystal::GetNbScatterer)
        .def("GetScatt", 
            (Scatterer& (Crystal::*)(const std::string&)) &Crystal::GetScatt, 
            return_internal_reference<>())
        .def("GetScatt", 
            (Scatterer& (Crystal::*)(const long)) &Crystal::GetScatt, 
            return_internal_reference<>())
        // FIXME - Needs converter.
        // I'm treating this the same way as GetSubObjRegistry from
        // refinableobj_ext.cpp.
        //.def("GetScattererRegistry", ( ObjRegistry<Scatterer>& 
        //    (Crystal::*) ()) &Crystal::GetScattererRegistry,
        //    return_internal_reference<>())
        //.def("GetScatteringPowerRegistry", ( ObjRegistry<ScatteringPower>& 
        //    (Crystal::*) ()) &Crystal::GetScatteringPowerRegistry,
        //    return_internal_reference<>())
        .def("AddScatteringPower", &Crystal::AddScatteringPower,
            with_custodian_and_ward<2,1, with_custodian_and_ward<1,2> >())
        // FIXME This deletes the ScatteringPower that is passed as an argument.
        //.def("RemoveScatteringPower", &Crystal::RemoveScatteringPower)
        .def("GetScatteringPower", 
            (ScatteringPower& (Crystal::*)(const std::string&)) 
            &Crystal::GetScatteringPower, 
            return_internal_reference<>())
        .def("GetMasterClockScatteringPower", &Crystal::GetMasterClockScatteringPower,
                return_value_policy<copy_const_reference>())
        .def("GetScatteringComponentList", 
            &Crystal::GetScatteringComponentList,
            return_internal_reference<>())
        .def("GetClockScattCompList", &Crystal::GetClockScattCompList,
                return_value_policy<copy_const_reference>())
        .def("GetMinDistanceTable", &Crystal::GetMinDistanceTable,
                (boost::python::arg("minDistance")=1.0))
        .def("PrintMinDistanceTable", &Crystal::PrintMinDistanceTable,
                (boost::python::arg("minDistance")=1.0, 
                 boost::python::arg("os")))
        .def("CalcDynPopCorr", &Crystal::CalcDynPopCorr,
                (boost::python::arg("overlapDist")=1.0, 
                 boost::python::arg("mergeDist")=0.0))
        .def("ResetDynPopCorr", &Crystal::ResetDynPopCorr)
        .def("SetUseDynPopCorr", &Crystal::SetUseDynPopCorr)
        .def("GetBumpMergeCost", &Crystal::GetBumpMergeCost)
        .def("SetBumpMergeDistance", 
            (void (Crystal::*)(const ScatteringPower&, const ScatteringPower&, const float))
             &Crystal::SetBumpMergeDistance,
             (boost::python::arg("scatt1"), 
              boost::python::arg("scatt2"), 
              boost::python::arg("dist")=1.5))
        .def("SetBumpMergeDistance", 
            (void (Crystal::*)
            (const ScatteringPower&, const ScatteringPower&, const float, const bool))
             &Crystal::SetBumpMergeDistance,
             (boost::python::arg("scatt1"), 
              boost::python::arg("scatt2"), 
              boost::python::arg("dist"), 
              boost::python::arg("allowMerge")))
        .def("RemoveBumpMergeDistance", &Crystal::RemoveBumpMergeDistance)
        // FIXME Need converter
        .def("GetBumpMergeParList", (Crystal::VBumpMergePar& (Crystal::*)())
            &Crystal::GetBumpMergeParList, return_internal_reference<>())
        .def("GetClockScattererList", &Crystal::GetClockScattererList,
                return_value_policy<copy_const_reference>())
        .def("CIFOutput", &Crystal::CIFOutput)
        .def("AddBondValenceRo", &Crystal::AddBondValenceRo)
        .def("RemoveBondValenceRo", &Crystal::AddBondValenceRo)
        .def("GetBondValenceCost", &Crystal::GetBondValenceCost)
        // FIXME Need converter
        .def("GetBondValenceRoList", 
            (std::map< pair< const ScatteringPower *, const ScatteringPower * >, float > &
            (Crystal::*)()) &Crystal::GetBondValenceRoList,
            return_internal_reference<>())
        ;


    class_<Crystal::BumpMergePar>("BumpMergePar", init<const float, optional<const bool> >())
        .def_readwrite("mDist2", &Crystal::BumpMergePar::mDist2)
        .def_readwrite("mCanOverlap", &Crystal::BumpMergePar::mCanOverlap)
        ;
}
