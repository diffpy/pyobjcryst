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
* Changes from ObjCryst++
* - The const-version of overloaded methods that can return either an internal
*   reference, or a constant interal reference are not wrapped.
* - CalcDynPopCorr is not enabled, as the API states that this is for internal
*   use only.
* - ResetDynPopCorr and SetUseDynPopCorr are not exposed, as these lead to
*   memory corruption in crystals that contain molecules.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Crystal.h"
#include "ObjCryst/UnitCell.h"
#include "ObjCryst/Atom.h"
#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

#include <string>
#include <map>
#include <set>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overloaded to protect scatterers from deletion.
void _AddScatterer(Crystal& crystal, Scatterer* scatt)
{
    crystal.AddScatterer(scatt);
    crystal.SetDeleteRefObjInDestructor(0);
    crystal.SetDeleteRefParInDestructor(0);
    // This is a workaround for a library bug
    crystal.SetUseDynPopCorr(0);
    return;
}

// Overloaded to protect scattering power from deletion.
void _AddScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    crystal.AddScatteringPower(scattpow);
    crystal.SetDeleteRefObjInDestructor(0);
    crystal.SetDeleteRefParInDestructor(0);
    // This is a workaround for a library bug
    crystal.SetUseDynPopCorr(0);
    return;
}

// Overloaded so that RemoveScatterer cannot delete the passed scatterer
void _RemoveScatterer(Crystal& crystal, Scatterer* scatt)
{
    crystal.RemoveScatterer(scatt, false);
}

// Overloaded so that RemoveScatteringPower cannot delete the passed
// scatteringpower 
void _RemoveScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    // Make sure that this scatteringpower is in this crystal

    crystal.RemoveScatteringPower(scattpow, false);
    return;
}

void _PrintMinDistanceTable(const Crystal& crystal, 
        const float minDistance = 0.1)
{

    crystal.PrintMinDistanceTable(minDistance);
    return;
}

// We want to turn a ScatteringComponentList into an actual list
bp::list _GetScatteringComponentList(Crystal &c)
{
    const ScatteringComponentList& scl = c.GetScatteringComponentList();
    bp::list l;
    for(size_t i = 0; i < scl.GetNbComponent(); ++i)
    {
        l.append(scl(i));
    }

    return l;
}

// wrap the virtual functions that need it
class CrystalWrap : public Crystal, public wrapper<Crystal>
{

    public: 

    CrystalWrap() : Crystal() {}

    CrystalWrap(const CrystalWrap& c) : Crystal(c) {}

    CrystalWrap(const float a, const float b, const float c , 
            const std::string& sg) 
        : Crystal(a, b, c, sg) {}
    CrystalWrap(const float a, const float b, const float c , 
            const float alpha, const float beta, const float gamma,
            const std::string& sg) 
        : Crystal(a, b, c, alpha, beta, gamma, sg) {}

    const ScatteringComponentList& default_GetScatteringComponentList() const
    { return this->Crystal::GetScatteringComponentList(); }

    const ScatteringComponentList& GetScatteringComponentList() const
    {
        if (override GetScatteringComponentList = 
                this->get_override("GetScatteringComponentList")) 
            return GetScatteringComponentList();
        return default_GetScatteringComponentList();
    }

};


} // namespace


BOOST_PYTHON_MODULE(_crystal)
{

    class_<CrystalWrap, bases<UnitCell>, boost::noncopyable > 
        ("Crystal", init<>())
        /* Constructors */
        .def(init<const float, const float, const float, const std::string&>())
        .def(init<const float, const float, const float, 
            const float, const float, const float, 
            const std::string&>())
        .def(init<const CrystalWrap&>())
        /* Methods */
        .def("AddScatterer", &_AddScatterer,
            with_custodian_and_ward<1,2>())
        .def("RemoveScatterer", &_RemoveScatterer)
        .def("GetNbScatterer", &Crystal::GetNbScatterer)
        .def("GetScatt", 
            (Scatterer& (Crystal::*)(const std::string&)) &Crystal::GetScatt, 
            return_internal_reference<>())
        .def("GetScatt", 
            (Scatterer& (Crystal::*)(const long)) &Crystal::GetScatt, 
            return_internal_reference<>())
        .def("GetScatterer", 
            (Scatterer& (Crystal::*)(const std::string&)) &Crystal::GetScatt, 
            return_internal_reference<>())
        .def("GetScatterer", 
            (Scatterer& (Crystal::*)(const long)) &Crystal::GetScatt, 
            return_internal_reference<>())
        .def("GetScattererRegistry", ( ObjRegistry<Scatterer>& 
            (Crystal::*) ()) &Crystal::GetScattererRegistry,
            return_internal_reference<>())
        .def("GetScatteringPowerRegistry", ( ObjRegistry<ScatteringPower>& 
            (Crystal::*) ()) &Crystal::GetScatteringPowerRegistry,
            return_internal_reference<>())
        .def("AddScatteringPower", &_AddScatteringPower,
            with_custodian_and_ward<1,2>())
        .def("RemoveScatteringPower", &_RemoveScatteringPower)
        .def("GetScatteringPower", 
            (ScatteringPower& (Crystal::*)(const std::string&)) 
            &Crystal::GetScatteringPower, 
            return_internal_reference<>())
        .def("GetMasterClockScatteringPower",
            &Crystal::GetMasterClockScatteringPower,
            return_value_policy<copy_const_reference>())
        .def("GetScatteringComponentList", &_GetScatteringComponentList,
            with_custodian_and_ward_postcall<1,0>())
        //.def("GetScatteringComponentList", 
        //    &CrystalWrap::GetScatteringComponentList,
        //    return_value_policy<copy_const_reference>())
        .def("GetClockScattCompList", &Crystal::GetClockScattCompList,
                return_value_policy<copy_const_reference>())
        .def("GetMinDistanceTable", &Crystal::GetMinDistanceTable,
                (bp::arg("minDistance")=1.0))
        .def("PrintMinDistanceTable", &_PrintMinDistanceTable,
                (bp::arg("minDistance")=1.0))
        //.def("CalcDynPopCorr", &Crystal::CalcDynPopCorr,
        //        (bp::arg("overlapDist")=1.0, 
        //         bp::arg("mergeDist")=0.0))
        .def("ResetDynPopCorr", &Crystal::ResetDynPopCorr)
        .def("SetUseDynPopCorr", &Crystal::SetUseDynPopCorr)
        .def("GetBumpMergeCost", &Crystal::GetBumpMergeCost)
        .def("SetBumpMergeDistance", 
            (void (Crystal::*)(const ScatteringPower&, const ScatteringPower&, const float))
             &Crystal::SetBumpMergeDistance,
             (bp::arg("scatt1"), 
              bp::arg("scatt2"), 
              bp::arg("dist")=1.5))
        .def("SetBumpMergeDistance", 
            (void (Crystal::*)
            (const ScatteringPower&, const ScatteringPower&, const float, const
             bool))
             &Crystal::SetBumpMergeDistance,
             (bp::arg("scatt1"), 
              bp::arg("scatt2"), 
              bp::arg("dist"), 
              bp::arg("allowMerge")))
        .def("RemoveBumpMergeDistance", &Crystal::RemoveBumpMergeDistance)
        .def("GetBumpMergeParList", (Crystal::VBumpMergePar& (Crystal::*)())
            &Crystal::GetBumpMergeParList, return_internal_reference<>())
        .def("GetClockScattererList", &Crystal::GetClockScattererList,
                return_value_policy<copy_const_reference>())
        .def("CIFOutput", &Crystal::CIFOutput)
        .def("AddBondValenceRo", &Crystal::AddBondValenceRo)
        .def("RemoveBondValenceRo", &Crystal::AddBondValenceRo)
        .def("GetBondValenceCost", &Crystal::GetBondValenceCost)
        .def("GetBondValenceRoList", 
            (std::map< pair< const ScatteringPower *, const ScatteringPower * >, float > &
            (Crystal::*)()) &Crystal::GetBondValenceRoList,
            return_internal_reference<>())
        ;


    class_<Crystal::BumpMergePar>("BumpMergePar", no_init)
        .def_readwrite("mDist2", &Crystal::BumpMergePar::mDist2)
        .def_readwrite("mCanOverlap", &Crystal::BumpMergePar::mCanOverlap)
        ;
}
