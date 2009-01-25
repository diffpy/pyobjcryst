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
* - The "Remove" functions do not delete the internal c++ objects. Care must be
*   taken to not use Scatterer and ScatteringPower python objects in multiple
*   crystals, which is in line with c++ counterparts. Here's why...
*
*   Crystal takes ownership of Scatterer and ScatteringPower objects and deletes
*   them in the destructor and in the "Remove" functions. Ownership can be
*   managed by assuring that the lifetime of the crystal is linked to that of
*   the sub-objects. In otherwords, we can't delete the crystal and then use the
*   the sub-objects. Likewise, we can't delete the sub-objects and expect them
*   to work in the crystal.  The "Remove" functions mess this up and can lead to
*   memory corruption when a user tries to access an object whose memory has
*   been deleted out from under it. Therefore, these functions aren't allowed to
*   delete their sub-objects.
*
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
#include <map>
#include <set>

using namespace boost::python;
using namespace ObjCryst;

namespace {

/* Overloaded so that RemoveScatterer does not delete the passed scatterer 
class CrystalWrap : public Crystal
{

    public:

    CrystalWrap() : Crystal() {};

    CrystalWrap(const float a, const float b, const float c,
        const std::string &SpaceGroupId) :
        Crystal(a, b, c, SpaceGroupId) {}

    CrystalWrap(const float a, const float b, const float c, 
        const float alpha, const float beta, const float gamma,
        const std::string &SpaceGroupId) :
        Crystal(a, b, c, alpha, beta, gamma, SpaceGroupId) {}

    CrystalWrap(const Crystal &oldCryst): Crystal(oldCryst) {}

    void RemoveScatterer(Scatterer *scatt)
    {
        
        ObjRegistry<Scatterer> &sr 
            = this->GetScattererRegistry();
        sr.DeRegister(*scatt);
        scatt->DeRegisterClient(*this);

        this->RemoveSubRefObj(*scatt);
        // This is what we don't want!
        // delete scatt;

        // this is ugly and cheating!
        RefinableObjClock& csl = 
            const_cast<RefinableObjClock&>(this->GetClockScattererList());
        csl.Print();
        csl.Click();
        csl.Print();
        return;
    }

    void Crystal::RemoveScatteringPower(ScatteringPower *scattPow)
    {

        ObjRegistry<ScatteringPower>& spr =
            this->GetScatteringPowerRegistry();
        spr.DeRegister(*scattPow);

        this->RemoveSubRefObj(*scattPow);
        mClockMaster.RemoveChild(scattPow->GetClockMaster());
        mClockMaster.RemoveChild(scattPow->GetMaximumLikelihoodParClock());

        RefinableObjClock& mcsp = 
            const_cast<RefinableObjClock&>
            (this->GetMasterClockScatteringPower());
        mcsp.RemoveChild(scattPow->GetClockMaster());
        //delete scattPow;

        VBumpMergePar& bmp = this->GetBumpMergeParList();
       
        for(Crystal::VBumpMergePar::iterator pos=mvBumpMergePar.begin();
            pos!=mvBumpMergePar.end();)
        {
            if((pos->first.first==scattPow)||(pos->first.second==scattPow))
            {
                bmp.erase(pos++);
                // FIXME
                //mBumpMergeParClock.Click();
            }
            else ++pos;// See Josuttis Std C++ Lib p.205 for safe method
        }


        std::map<pair<const ScatteringPower*,const ScatteringPower*>, float>&  
            mvBondValenceRo = this->GetBondValenceRoList();
       
        for(map<pair<const ScatteringPower*,const ScatteringPower*>, float>::iterator
            pos=mvBondValenceRo.begin();pos!=mvBondValenceRo.end();)
        {
        if((pos->first.first==scattPow)||(pos->first.second==scattPow))
            {
                mvBondValenceRo.erase(pos++);
                //FIXME
                //mBondValenceParClock.Click();
            }
            else ++pos;
        }
    }

};
*/

/* We want to make sure that a scatterer doesn't get added to more than one
 * crystal.
 */
std::set<Scatterer*> scattreg;

void _AddScatterer(Crystal& crystal, Scatterer* scatt)
{
    if( scattreg.count(scatt) > 0 )
    {
        PyErr_SetString(PyExc_AttributeError, "Scatterer already belongs to a crystal.");
        throw_error_already_set();
        return;
    }
    scattreg.insert(scatt);
    crystal.AddScatterer(scatt);
    return;
}

std::set<ScatteringPower*> scattpowreg;

void _AddScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    if( scattpowreg.count(scattpow) > 0 )
    {
        PyErr_SetString(PyExc_AttributeError, "ScatteringPower already belongs to a crystal.");
        throw_error_already_set();
        return;
    }
    scattpowreg.insert(scattpow);
    crystal.AddScatteringPower(scattpow);
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
        //.def("RemoveScatterer", &Crystal::RemoveScatterer)
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
