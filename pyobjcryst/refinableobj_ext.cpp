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
* boost::python bindings to ObjCryst::RefinableObj. This is a virtual class that
* can be derived from in python. These bindings are used by ObjCryst objects
* that inherit from RefinableObj (see for example unitcell_ext.cpp).
* RefinableObj derivatives can be created in python and will work in c++
* functions that are also bound into python.
* 
* Changes from ObjCryst++
* - GetPar that takes a const float* is not exposed, as it is designed for
*   internal use.
* - GetParamSet returns a copy of the internal data so that no indirect
*   manipulation can take place from python.
* - SetDeleteRefParInDestructor(false) is called in the constructors of the python
*   class.
* - SetDeleteRefParInDestructor is not exposed.
* - RemovePar is overloaded to return None.
* - XMLOutput and XMLInput are not wrapped (yet).
*
* $Id$
*
*****************************************************************************/

#include <string>
#include <map>
#include <iostream>

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/to_python_converter.hpp>

#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

#include "helpers.h"

using namespace ObjCryst;
using namespace boost::python;

namespace {

class RefinableObjWrap : public RefinableObj, 
                 public wrapper<RefinableObj>
{

    public: 

    /* alias the constructors from RefinableObj */
    RefinableObjWrap() : RefinableObj() 
    {
        RefinableObj::SetDeleteRefParInDestructor(false);
    }
    RefinableObjWrap(const bool internal) : RefinableObj(internal)
    {
        RefinableObj::SetDeleteRefParInDestructor(false);
    }

    // Fix for const void issue
    void EraseAllParamSet() {
        this->RefinableObj::EraseAllParamSet();
    }

    const std::string& default_GetClassName() const
    { return this->RefinableObj::GetClassName(); }

    const std::string& GetClassName() const
    {
        if (override GetClassName = this->get_override("GetClassName")) 
            return GetClassName();
        return default_GetClassName();
    }

    const std::string& default_GetName() const
    { return this->RefinableObj::GetName(); }

    const std::string& GetName() const
    {
        if (override GetName = this->get_override("GetName")) 
            return GetName();
        return default_GetName();
    }

    void default_SetName(const std::string &name)
    { this->RefinableObj::SetName(name); }

    void SetName(const std::string &name)
    {
        if (override SetName = this->get_override("SetName")) 
            SetName(name);
        default_SetName(name);
    }

    void default_Print() const
    { this->RefinableObj::Print();}

    void Print() const
    {
        if (override Print = this->get_override("Print")) 
            Print();
        default_Print();
    }

    void default_RegisterClient(RefinableObj& client) const 
    { this->RefinableObj::RegisterClient(client); }

    void RegisterClient(RefinableObj& client) const
    {
        if (override RegisterClient = this->get_override("RegisterClient")) 
            RegisterClient(client);
        default_RegisterClient(client);
    }

    void default_DeRegisterClient(RefinableObj& client) const 
    { this->RefinableObj::DeRegisterClient(client); }

    void DeRegisterClient(RefinableObj& client) const
    {
        if (override DeRegisterClient = this->get_override("DeRegisterClient")) 
            DeRegisterClient(client);
        default_DeRegisterClient(client);
    }

    ObjRegistry< RefinableObj >& default_GetClientRegistry()
    { return this->RefinableObj::GetClientRegistry(); }

    ObjRegistry< RefinableObj >& GetClientRegistry()
    {
        if (override GetClientRegistry = this->get_override("GetClientRegistry")) 
            return GetClientRegistry();
        return default_GetClientRegistry();
    }

    void default_BeginOptimization(const bool allowApproximations, 
            const bool enableRestraints) 
    { this->RefinableObj::BeginOptimization(allowApproximations, enableRestraints); }

    void BeginOptimization(const bool allowApproximations, 
            const bool enableRestraints) 
    {
        if (override BeginOptimization = this->get_override("BeginOptimization")) 
            BeginOptimization(allowApproximations, enableRestraints);
        default_BeginOptimization(allowApproximations, enableRestraints);
    }

    void default_EndOptimization()
    { this->RefinableObj::EndOptimization();}

    void EndOptimization()
    {
        if (override EndOptimization = this->get_override("EndOptimization")) 
            EndOptimization();
        default_EndOptimization();
    }

    void default_RandomizeConfiguration()
    { this->RefinableObj::RandomizeConfiguration();}

    void RandomizeConfiguration()
    {
        if (override RandomizeConfiguration = 
                this->get_override("RandomizeConfiguration")) 
            RandomizeConfiguration();
        default_RandomizeConfiguration();
    }

    void default_GlobalOptRandomMove(const float mutationAmplitude,
            const RefParType *type)
    { this->RefinableObj::GlobalOptRandomMove(mutationAmplitude, type);}

    void GlobalOptRandomMove(const float mutationAmplitude,
            const RefParType *type)
    {
        if (override GlobalOptRandomMove = this->get_override("GlobalOptRandomMove")) 
            GlobalOptRandomMove(mutationAmplitude, type);
        default_GlobalOptRandomMove(mutationAmplitude, type);
    }

    float default_GetLogLikelihood() const
    { return this->RefinableObj::GetLogLikelihood(); }

    float GetLogLikelihood() const
    {
        if (override GetLogLikelihood = this->get_override("GetLogLikelihood")) 
            return GetLogLikelihood();
        return default_GetLogLikelihood();
    }

    unsigned int default_GetNbLSQFunction() const
    { return this->RefinableObj::GetNbLSQFunction(); }

    unsigned int GetNbLSQFunction() const
    {
        if (override GetNbLSQFunction = this->get_override("GetNbLSQFunction")) 
            return GetNbLSQFunction();
        return default_GetNbLSQFunction();
    }

    const CrystVector<float>& default_GetLSQCalc(const unsigned int i) const 
    { return this->RefinableObj::GetLSQCalc(i); }

    const CrystVector<float>& GetLSQCalc(const unsigned int i) const 
    {
        if (override GetLSQCalc = this->get_override("GetLSQCalc")) 
            return GetLSQCalc(i);
        return default_GetLSQCalc(i);
    }

    const CrystVector<float>& default_GetLSQObs(const unsigned int i) const 
    { return this->RefinableObj::GetLSQObs(i); }

    const CrystVector<float>& GetLSQObs(const unsigned int i) const 
    {
        if (override GetLSQObs = this->get_override("GetLSQObs")) 
            return GetLSQObs(i);
        return default_GetLSQObs(i);
    }

    const CrystVector<float>& default_GetLSQWeight(const unsigned int i) const 
    { return this->RefinableObj::GetLSQWeight(i); }

    const CrystVector<float>& GetLSQWeight(const unsigned int i) const 
    {
        if (override GetLSQWeight = this->get_override("GetLSQWeight")) 
            return GetLSQWeight(i);
        return default_GetLSQWeight(i);
    }

    const CrystVector<float>& default_GetLSQDeriv(const unsigned int i,
            RefinablePar &rp)
    { return this->RefinableObj::GetLSQDeriv(i, rp); }

    const CrystVector<float>& GetLSQDeriv(const unsigned int i,
            RefinablePar &rp)
    {
        if (override GetLSQDeriv = this->get_override("GetLSQDeriv")) 
            return GetLSQDeriv(i, rp);
        return default_GetLSQDeriv(i, rp);
    }

    void default_XMLOutput(std::ostream &os, int indent) const
    { this->RefinableObj::XMLOutput(os, indent); }

    void XMLOutput(std::ostream &os, int indent) const
    {
        if (override XMLOutput = this->get_override("XMLOutput")) 
            XMLOutput(os, indent);
        default_XMLOutput(os, indent);
    }

    void default_XMLInput(std::istream &is, 
            const ObjCryst::XMLCrystTag &tag)
    { this->RefinableObj::XMLInput(is, tag); }

    void XMLInput(std::istream &is, 
            const ObjCryst::XMLCrystTag &tag)
    {
        if (override XMLInput = this->get_override("XMLInput")) 
            XMLInput(is, tag);
        default_XMLInput(is, tag);
    }

    void default_GetGeneGroup(const ObjCryst::RefinableObj &obj,
            CrystVector<unsigned int> &groupIndex,
            unsigned int &firstGroup) const
    { this->RefinableObj::GetGeneGroup(obj, groupIndex, firstGroup);}

    void GetGeneGroup(const ObjCryst::RefinableObj &obj,
            CrystVector<unsigned int> &groupIndex,
            unsigned int &firstGroup) const
    {
        if (override GetGeneGroup = this->get_override("GetGeneGroup")) 
            GetGeneGroup(obj, groupIndex, firstGroup);
        default_GetGeneGroup(obj, groupIndex, firstGroup);
    }

    void default_UpdateDisplay() const
    { this->RefinableObj::UpdateDisplay();}

    void UpdateDisplay() const
    {
        if (override UpdateDisplay = this->get_override("UpdateDisplay")) 
            UpdateDisplay();
        default_UpdateDisplay();
    }

    float default_GetRestraintCost() const
    { return this->RefinableObj::GetRestraintCost();}

    float GetRestraintCost() const
    {
        if (override GetRestraintCost = this->get_override("GetRestraintCost")) 
            return GetRestraintCost();
        return default_GetRestraintCost();
    }

    void default_TagNewBestConfig() const
    { this->RefinableObj::TagNewBestConfig();}

    void TagNewBestConfig() const
    {
        if (override TagNewBestConfig = this->get_override("TagNewBestConfig")) 
            TagNewBestConfig();
        default_TagNewBestConfig();
    }

    // Protected methods

    void default_Prepare() 
    { RefinableObj::Prepare();}
    
    void Prepare() 
    {
        if (override Prepare = this->get_override("Prepare")) 
            Prepare();
        default_Prepare();
    }

    long FindPar(const std::string& name) const
    { return RefinableObj::FindPar(name); }

    void AddOption(RefObjOpt *opt)
    { RefinableObj::AddOption(opt); }

    std::map<unsigned long, 
        std::pair<
            CrystVector<float>, std::string>
        >::iterator
        FindParamSet(unsigned long id) const
        { return RefinableObj::FindParamSet(id); }

};

void _RemovePar(RefinableObj &obj, RefinablePar* refpar)
{
    obj.RemovePar(refpar);
    return;
}

} // anonymous namespace


BOOST_PYTHON_MODULE(_refinableobj)
{

    class_<RefinableObjWrap, boost::noncopyable>("RefinableObj")
        .def(init<bool>())
        // Defined not implemented
        //.def(init<const RefinableObj&>())
        /* Methods */
        .def("PrepareForRefinement", &RefinableObj::PrepareForRefinement)
        .def("FixAllPar", &RefinableObj::FixAllPar)
        .def("UnFixAllPar", &RefinableObj::UnFixAllPar)
        .def("SetParIsFixed", (void (RefinableObj::*)(const long, const bool))
            &RefinableObj::SetParIsFixed)
        .def("SetParIsFixed", (void (RefinableObj::*)(const std::string&, const bool))
            &RefinableObj::SetParIsFixed)
        .def("SetParIsFixed", (void (RefinableObj::*)(const RefParType*, const bool))
            &RefinableObj::SetParIsFixed)
        .def("SetParIsUsed", (void (RefinableObj::*)(const std::string&, const bool))
            &RefinableObj::SetParIsUsed)
        .def("SetParIsUsed", (void (RefinableObj::*)(const RefParType*, const bool))
            &RefinableObj::SetParIsUsed)
        .def("GetNbPar", &RefinableObj::GetNbPar)
        .def("GetNbParNotFixed", &RefinableObj::GetNbParNotFixed)
        .def("GetPar", (RefinablePar& (RefinableObj::*)(const long)) 
            &RefinableObj::GetPar,
            return_internal_reference<>())
        .def("GetPar", (RefinablePar& (RefinableObj::*)(const string&)) 
            &RefinableObj::GetPar,
            return_internal_reference<>())
        .def("GetParNotFixed", (RefinablePar& (RefinableObj::*)(const long)) 
            &RefinableObj::GetParNotFixed,
            return_internal_reference<>())
        .def("AddPar", (void (RefinableObj::*)(RefinablePar*)) 
            &RefinableObj::AddPar,
            with_custodian_and_ward<1,2>())
        .def("AddPar", (void (RefinableObj::*)(RefinableObj&, const bool)) 
            &RefinableObj::AddPar,
            with_custodian_and_ward<1,2>())
        .def("RemovePar", &_RemovePar)
        .def("CreateParamSet", &RefinableObj::CreateParamSet)
        .def("ClearParamSet", &RefinableObj::ClearParamSet)
        .def("SaveParamSet", &RefinableObj::SaveParamSet)
        .def("RestoreParamSet", &RefinableObj::RestoreParamSet)
        .def("GetParamSet", (const CrystVector<float>& (RefinableObj::*)
            (const unsigned long) const) &RefinableObj::GetParamSet,
            return_value_policy<copy_const_reference>())
        .def("GetParamSet_ParNotFixedHumanValue", 
            &RefinableObj::GetParamSet_ParNotFixedHumanValue)
        .def("EraseAllParamSet", &RefinableObjWrap::EraseAllParamSet)
        .def("GetParamSetName", &RefinableObj::GetParamSetName,
            return_value_policy<copy_const_reference>())
        .def("SetLimitsAbsolute", ( void (RefinableObj::*)
            (const std::string&, const float, const float) ) 
            &RefinableObj::SetLimitsAbsolute)
        .def("SetLimitsAbsolute", ( void (RefinableObj::*)
            (const RefParType*, const float, const float) ) 
            &RefinableObj::SetLimitsAbsolute)
        .def("SetLimitsRelative", ( void (RefinableObj::*)
            (const std::string&, const float, const float) ) 
            &RefinableObj::SetLimitsRelative)
        .def("SetLimitsRelative", ( void (RefinableObj::*)
            (const RefParType*, const float, const float) ) 
            &RefinableObj::SetLimitsRelative)
        .def("SetLimitsProportional", ( void (RefinableObj::*)
            (const std::string&, const float, const float) ) 
            &RefinableObj::SetLimitsProportional)
        .def("SetLimitsProportional", ( void (RefinableObj::*)
            (const RefParType*, const float, const float) ) 
            &RefinableObj::SetLimitsProportional)
        .def("SetGlobalOptimStep", &RefinableObj::SetGlobalOptimStep)
        .def("GetSubObjRegistry", ( ObjRegistry<RefinableObj>& 
            (RefinableObj::*) ()) &RefinableObj::GetSubObjRegistry,
            return_internal_reference<>())
        .def("IsBeingRefined", &RefinableObj::IsBeingRefined)
        .def("BeginGlobalOptRandomMove", 
            &RefinableObj::BeginGlobalOptRandomMove)
        .def("ResetParList", &RefinableObj::ResetParList)
        .def("GetNbOption", &RefinableObj::GetNbOption)
        .def("GetOption", (RefObjOpt& (RefinableObj::*)(const unsigned int))
            &RefinableObj::GetOption,
            return_internal_reference<>())
        // Not exposed, as we want python to manage the objects
        //.def("SetDeleteRefParInDestructor", 
        //    &RefinableObj::SetDeleteRefParInDestructor)
        .def("GetRefParListClock", &RefinableObj::GetRefParListClock,
            return_value_policy<copy_const_reference>())
        .def("AddRestraint", &RefinableObj::AddRestraint,
            with_custodian_and_ward<2,1,with_custodian_and_ward<1,2> >())
        .def("RemoveRestraint", &RefinableObj::RemoveRestraint)
        .def("GetClockMaster", &RefinableObj::GetClockMaster,
            return_value_policy<copy_const_reference>())
        // Virtual
        .def("GetClassName", &RefinableObj::GetClassName, 
            &RefinableObjWrap::default_GetClassName,
            return_value_policy<copy_const_reference>())
        .def("GetName", &RefinableObj::GetName, 
            &RefinableObjWrap::default_GetName,
            return_value_policy<copy_const_reference>())
        .def("SetName", &RefinableObj::SetName, 
            &RefinableObjWrap::default_SetName)
        .def("Print", &RefinableObj::Print, 
            &RefinableObjWrap::default_Print)
        .def("RegisterClient", &RefinableObj::RegisterClient, 
            &RefinableObjWrap::default_RegisterClient,
            with_custodian_and_ward<1,2>())
        .def("DeRegisterClient", &RefinableObj::DeRegisterClient, 
            &RefinableObjWrap::default_DeRegisterClient)
        .def("GetClientRegistry", ( ObjRegistry<RefinableObj>& 
            (RefinableObj::*) ()) &RefinableObj::GetClientRegistry,
            &RefinableObjWrap::default_GetClientRegistry,
            return_internal_reference<>())
        .def("BeginOptimization", &RefinableObj::BeginOptimization, 
            &RefinableObjWrap::default_BeginOptimization)
        .def("EndOptimization", &RefinableObj::EndOptimization, 
            &RefinableObjWrap::default_EndOptimization)
        .def("RandomizeConfiguration", &RefinableObj::RandomizeConfiguration, 
            &RefinableObjWrap::default_RandomizeConfiguration)
        .def("GlobalOptRandomMove", &RefinableObj::GlobalOptRandomMove, 
            &RefinableObjWrap::default_GlobalOptRandomMove)
        .def("GetLogLikelihood", &RefinableObj::GetLogLikelihood, 
            &RefinableObjWrap::default_GetLogLikelihood)
        .def("GetNbLSQFunction", &RefinableObj::GetNbLSQFunction, 
            &RefinableObjWrap::default_GetNbLSQFunction)
        .def("GetLSQCalc", &RefinableObj::GetLSQCalc, 
            &RefinableObjWrap::default_GetLSQCalc,
            return_value_policy<copy_const_reference>())
        .def("GetLSQObs", &RefinableObj::GetLSQObs, 
            &RefinableObjWrap::default_GetLSQObs,
            return_value_policy<copy_const_reference>())
        .def("GetLSQWeight", &RefinableObj::GetLSQWeight, 
            &RefinableObjWrap::default_GetLSQWeight,
            return_value_policy<copy_const_reference>())
        .def("GetLSQDeriv", &RefinableObj::GetLSQDeriv, 
            &RefinableObjWrap::default_GetLSQDeriv,
            return_value_policy<copy_const_reference>())
        // FIXME - Need to deal with the ostream argument
        //.def("XMLOutput", &RefinableObj::XMLOutput, 
        //    &RefinableObjWrap::default_XMLOutput)
        // FIXME - Need to deal with the istream argument
        //.def("XMLInput", &RefinableObj::XMLInput, 
        //    &RefinableObjWrap::default_XMLInput)
        .def("GetGeneGroup", &RefinableObj::GetGeneGroup, 
            &RefinableObjWrap::default_GetGeneGroup)
        .def("GetRestraintCost", &RefinableObj::GetRestraintCost, 
            &RefinableObjWrap::default_GetRestraintCost)
        .def("TagNewBestConfig", &RefinableObj::TagNewBestConfig, 
            &RefinableObjWrap::default_TagNewBestConfig)
        // Additional methods for python only
        .def("__str__", &__str__<RefinableObj>)
        ;
}
