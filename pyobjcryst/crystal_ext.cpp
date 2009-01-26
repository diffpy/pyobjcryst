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
* boost::python bindings to ObjCryst::Crystal.  * * Changes from ObjCryst++
* - The const-version of overloaded methods that can return either an internal
*   reference, or a constant interal reference are not wrapped.
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

using namespace boost::python;
using namespace ObjCryst;

namespace {

/* We want to make sure that Scatterers and ScatterinPowers don't get added to
 * more than one crystal and have a unique name in a given crystal. These help
 * manage so that it is fast.
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

// Overloaded to record ScatterinPower in the registry
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

    std::set<Scatterer*>::iterator siter;
    siter = scattreg[&crystal].find(scatt);
    if(siter != scattreg[&crystal].end() )
    {
        crystal.RemoveScatterer(scatt, false);
        scattreg[&crystal].erase( *siter );
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

    std::set<ScatteringPower*>::iterator siter;
    siter = scattpowreg[&crystal].find(scattpow);
    if(siter != scattpowreg[&crystal].end() )
    {
        crystal.RemoveScatteringPower(scattpow, false);
        scattpowreg[&crystal].erase( *siter );
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


} // namespace


BOOST_PYTHON_MODULE(_crystal)
{

    class_<Crystal, bases<UnitCell> > ("Crystal", init<>())
        /* Constructors */
        .def(init<const float, const float, const float, const std::string&>())
        .def(init<const float, const float, const float, 
            const float, const float, const float, 
            const std::string&>())
        .def(init<const Crystal&>())
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
        // FIXME - Needs converter.
        // I'm treating this the same way as GetSubObjRegistry from
        // refinableobj_ext.cpp.
        //.def("GetScattererRegistry", ( ObjRegistry<Scatterer>& 
        //    (Crystal::*) ()) &Crystal::GetScattererRegistry,
        //    return_internal_reference<>())
        //.def("GetScatteringPowerRegistry", ( ObjRegistry<ScatteringPower>& 
        //    (Crystal::*) ()) &Crystal::GetScatteringPowerRegistry,
        //    return_internal_reference<>())
        .def("AddScatteringPower", &_AddScatteringPower,
            with_custodian_and_ward<2,1,with_custodian_and_ward<1,2> >())
        .def("RemoveScatteringPower", &_RemoveScatteringPower)
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


    class_<Crystal::BumpMergePar>("BumpMergePar", init<const float, optional<const bool> >())
        .def_readwrite("mDist2", &Crystal::BumpMergePar::mDist2)
        .def_readwrite("mCanOverlap", &Crystal::BumpMergePar::mCanOverlap)
        ;
}
