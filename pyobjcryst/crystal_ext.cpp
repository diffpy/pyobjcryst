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
#include <sstream>
#include <map>
#include <set>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

/* We want to make sure that Scatterers and ScatterinPowers don't get added to
 * more than one crystal and have a unique name in a given crystal. These help
 * manage this.
 */
std::map<Crystal*, std::set<Scatterer*> > scattreg;
std::map<Crystal*, std::set<string> > scattnamereg;
std::map<Crystal*, std::set<ScatteringPower*> > scattpowreg;
std::map<Crystal*, std::set<string> > scattpownamereg;

// Overloaded to record Scatterer in the registry
void _AddScatterer(Crystal& crystal, Scatterer* scatt)
{
    // Make sure that the scatterer would have a unique name in this crystal
    if( scattnamereg[&crystal].count(scatt->GetName()) > 0 )
    {
        std::stringstream ss;
        ss << "Crystal already has Scatterer with name '"; 
        ss << scatt->GetName() << "'";
        PyErr_SetString(PyExc_AttributeError, ss.str().c_str());
        throw_error_already_set();
        return;
    }
    // Make sure the scatterer isn't already in a crystal
    std::map<Crystal*, std::set<Scatterer*> >::iterator citer;
    for( citer = scattreg.begin(); citer != scattreg.end(); ++citer)
    {
        if( (citer->second).count(scatt) > 0 )
        {
            std::stringstream ss;
            ss << "Scatterer '" << scatt->GetName(); 
            ss << "' already belongs to a crystal.";
            PyErr_SetString(PyExc_AttributeError, ss.str().c_str());
            throw_error_already_set();
            return;
        }
    }
    // If we got here, then we're ok
    scattreg[&crystal].insert( scatt );
    scattnamereg[&crystal].insert( scatt->GetName() );
    crystal.AddScatterer(scatt);
    return;
}

// Overloaded to record ScatteringPower in the registry
void _AddScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    // Make sure that the scatteringpower would have a unique name in this
    // crystal
    if( scattpownamereg[&crystal].count(scattpow->GetName()) > 0 )
    {
        std::stringstream ss;
        ss << "Crystal already has ScatteringPower with name '"; 
        ss << scattpow->GetName() << "'";
        PyErr_SetString(PyExc_AttributeError, ss.str().c_str());
        throw_error_already_set();
        return;
    }
    // Make sure the scatteringpower isn't already in a crystal
    std::map<Crystal*, std::set<ScatteringPower*> >::iterator citer;
    for( citer = scattpowreg.begin(); citer != scattpowreg.end(); ++citer)
    {
        if( (citer->second).count(scattpow) > 0)
        {
            std::stringstream ss;
            ss << "ScatteringPower '" << scattpow->GetName(); 
            ss << "' already belongs to a crystal.";
            PyErr_SetString(PyExc_AttributeError, ss.str().c_str());
            throw_error_already_set();
            return;
        }
    }
    // If we got here, then we're ok
    scattpowreg[&crystal].insert( scattpow );
    scattpownamereg[&crystal].insert( scattpow->GetName() );
    crystal.AddScatteringPower(scattpow);
    return;
}

// Overloaded so that RemoveScatterer cannot delete the passed scatterer, and so
// that one cannot remove a scatterer that is not in the crystal
void _RemoveScatterer(Crystal& crystal, Scatterer* scatt)
{
    // Make sure that this scatterer is in this crystal

    if(scattreg[&crystal].count( scatt) > 0 )
    {
        crystal.RemoveScatterer(scatt, false);
        scattnamereg[&crystal].erase( scatt->GetName() );
        scattreg[&crystal].erase( scatt );
    }
    else
    {
        std::stringstream ss;
        ss << "Scatterer '" << scatt->GetName(); 
        ss << "' does not belong to crystal.";
        PyErr_SetString(PyExc_AttributeError, ss.str().c_str());
        throw_error_already_set();
    }
    return;
}

// Overloaded so that RemoveScatteringPower cannot delete the passed
// scatteringpower 
void _RemoveScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    // Make sure that this scatteringpower is in this crystal

    if(scattpowreg[&crystal].count(scattpow) > 0 )
    {
        crystal.RemoveScatteringPower(scattpow, false);
        scattpownamereg[&crystal].erase( scattpow->GetName() );
        scattpowreg[&crystal].erase( scattpow );
    }
    else
    {
        std::stringstream ss;
        ss << "ScatteringPower '" << scattpow->GetName(); 
        ss << "' does not belong to crystal.";
        PyErr_SetString(PyExc_AttributeError, ss.str().c_str());
        throw_error_already_set();
    }
    return;
}

void _PrintMinDistanceTable(const Crystal& crystal, 
        const float minDistance = 0.1)
{

    crystal.PrintMinDistanceTable(minDistance);
    return;
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
            with_custodian_and_ward<2,1,with_custodian_and_ward<1,2> >())
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
            with_custodian_and_ward<2,1,with_custodian_and_ward<1,2> >())
        .def("RemoveScatteringPower", &_RemoveScatteringPower)
        .def("GetScatteringPower", 
            (ScatteringPower& (Crystal::*)(const std::string&)) 
            &Crystal::GetScatteringPower, 
            return_internal_reference<>())
        .def("GetMasterClockScatteringPower",
            &Crystal::GetMasterClockScatteringPower,
            return_value_policy<copy_const_reference>())
        .def("GetScatteringComponentList", 
            &CrystalWrap::GetScatteringComponentList,
            return_value_policy<copy_const_reference>())
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
