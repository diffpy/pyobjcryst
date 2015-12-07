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
* boost::python bindings to ObjCryst::RefinableObj. This is a virtual class that
* can be derived from in python. These bindings are used by ObjCryst objects
* that inherit from RefinableObj (see for example unitcell_ext.cpp).
* RefinableObj derivatives can be created in python and will work in c++
* functions that are also bound into python.
*
* Changes from ObjCryst::RefinableObj
* - XMLOutput and XMLInput accept python file-like objects.
* - GetPar that takes a const double* is not exposed, as it is designed for
*   internal use.
* - GetParamSet returns a copy of the internal data so that no indirect
*   manipulation can take place from python.
* - SetDeleteRefParInDestructor(false) is called in the constructors of the
*   python class and the parameter accessors.
* - SetDeleteRefParInDestructor is not exposed.
* - RemovePar is overloaded to return None.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>
#include <map>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers.hpp"
#include "python_streambuf.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {


// Workaround to SetDeleteRefParInDestructor(0) when a parameter is added

void _AddPar(RefinableObj& obj, RefinablePar* p)
{
    obj.AddPar(p);
    obj.SetDeleteRefParInDestructor(0);
}

void _AddParObj(RefinableObj& obj, RefinableObj& o, const bool copyParam = false)
{
    obj.AddPar(o, copyParam);
    obj.SetDeleteRefParInDestructor(0);
}

RefinablePar& _GetParLong(RefinableObj& obj, const long i)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetPar(i);
}

RefinablePar& _GetParString(RefinableObj& obj, const string& s)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetPar(s);
}

RefinablePar& _GetParNotFixedLong(RefinableObj& obj, const long i)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetParNotFixed(i);
}


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
        override f = this->get_override("GetClassName");
        if (f)  return f();
        return default_GetClassName();
    }

    const std::string& default_GetName() const
    { return this->RefinableObj::GetName(); }

    const std::string& GetName() const
    {
        override f = this->get_override("GetName");
        if (f)  return f();
        return default_GetName();
    }

    void default_SetName(const std::string& name)
    { this->RefinableObj::SetName(name); }

    void SetName(const std::string& name)
    {
        override f = this->get_override("SetName");
        if (f)  f(name);
        else  default_SetName(name);
    }

    void default_Print() const
    { this->RefinableObj::Print();}

    void Print() const
    {
        override f = this->get_override("Print");
        if (f)  f();
        else  default_Print();
    }

    void default_RegisterClient(RefinableObj& client) const
    { this->RefinableObj::RegisterClient(client); }

    void RegisterClient(RefinableObj& client) const
    {
        override f = this->get_override("RegisterClient");
        if (f)  f(client);
        else  default_RegisterClient(client);
    }

    void default_DeRegisterClient(RefinableObj& client) const
    { this->RefinableObj::DeRegisterClient(client); }

    void DeRegisterClient(RefinableObj& client) const
    {
        override f = this->get_override("DeRegisterClient");
        if (f)  f(client);
        else  default_DeRegisterClient(client);
    }

    ObjRegistry< RefinableObj >& default_GetClientRegistry()
    { return this->RefinableObj::GetClientRegistry(); }

    ObjRegistry< RefinableObj >& GetClientRegistry()
    {
        override f = this->get_override("GetClientRegistry");
        if (f)  return f();
        return default_GetClientRegistry();
    }

    void default_BeginOptimization(const bool allowApproximations,
            const bool enableRestraints)
    { this->RefinableObj::BeginOptimization(allowApproximations, enableRestraints); }

    void BeginOptimization(const bool allowApproximations,
            const bool enableRestraints)
    {
        override f = this->get_override("BeginOptimization");
        if (f)  f(allowApproximations, enableRestraints);
        else  default_BeginOptimization(allowApproximations, enableRestraints);
    }

    void default_EndOptimization()
    { this->RefinableObj::EndOptimization();}

    void EndOptimization()
    {
        override f = this->get_override("EndOptimization");
        if (f)  f();
        else  default_EndOptimization();
    }

    void default_RandomizeConfiguration()
    { this->RefinableObj::RandomizeConfiguration();}

    void RandomizeConfiguration()
    {
        override f = this->get_override("RandomizeConfiguration");
        if (f)  f();
        else  default_RandomizeConfiguration();
    }

    void default_GlobalOptRandomMove(const double mutationAmplitude,
            const RefParType* type)
    { this->RefinableObj::GlobalOptRandomMove(mutationAmplitude, type);}

    void GlobalOptRandomMove(const double mutationAmplitude,
            const RefParType* type)
    {
        override f = this->get_override("GlobalOptRandomMove");
        if (f)  f(mutationAmplitude, type);
        else  default_GlobalOptRandomMove(mutationAmplitude, type);
    }

    double default_GetLogLikelihood() const
    { return this->RefinableObj::GetLogLikelihood(); }

    double GetLogLikelihood() const
    {
        override f = this->get_override("GetLogLikelihood");
        if (f)  return f();
        return default_GetLogLikelihood();
    }

    unsigned int default_GetNbLSQFunction() const
    { return this->RefinableObj::GetNbLSQFunction(); }

    unsigned int GetNbLSQFunction() const
    {
        override f = this->get_override("GetNbLSQFunction");
        if (f)  return f();
        return default_GetNbLSQFunction();
    }

    const CrystVector<double>& default_GetLSQCalc(const unsigned int i) const
    { return this->RefinableObj::GetLSQCalc(i); }

    const CrystVector<double>& GetLSQCalc(const unsigned int i) const
    {
        override f = this->get_override("GetLSQCalc");
        if (f)  return f(i);
        return default_GetLSQCalc(i);
    }

    const CrystVector<double>& default_GetLSQObs(const unsigned int i) const
    { return this->RefinableObj::GetLSQObs(i); }

    const CrystVector<double>& GetLSQObs(const unsigned int i) const
    {
        override f = this->get_override("GetLSQObs");
        if (f)  return f(i);
        return default_GetLSQObs(i);
    }

    const CrystVector<double>& default_GetLSQWeight(const unsigned int i) const
    { return this->RefinableObj::GetLSQWeight(i); }

    const CrystVector<double>& GetLSQWeight(const unsigned int i) const
    {
        override f = this->get_override("GetLSQWeight");
        if (f)  return f(i);
        return default_GetLSQWeight(i);
    }

    const CrystVector<double>& default_GetLSQDeriv(const unsigned int i,
            RefinablePar& rp)
    { return this->RefinableObj::GetLSQDeriv(i, rp); }

    const CrystVector<double>& GetLSQDeriv(const unsigned int i,
            RefinablePar& rp)
    {
        override f = this->get_override("GetLSQDeriv");
        if (f)  return f(i, rp);
        return default_GetLSQDeriv(i, rp);
    }

    void default_XMLOutput(std::ostream& os, int indent) const
    { this->RefinableObj::XMLOutput(os, indent); }

    void XMLOutput(std::ostream& os, int indent) const
    {
        override f = this->get_override("XMLOutput");
        if (f)  f(os, indent);
        else  default_XMLOutput(os, indent);
    }

    void default_XMLInput(std::istream& is,
            const ObjCryst::XMLCrystTag& tag)
    { this->RefinableObj::XMLInput(is, tag); }

    void XMLInput(std::istream& is,
            const ObjCryst::XMLCrystTag& tag)
    {
        override f = this->get_override("XMLInput");
        if (f)  f(is, tag);
        else  default_XMLInput(is, tag);
    }

    void default_GetGeneGroup(const ObjCryst::RefinableObj& obj,
            CrystVector<unsigned int>& groupIndex,
            unsigned int& firstGroup) const
    { this->RefinableObj::GetGeneGroup(obj, groupIndex, firstGroup);}

    void GetGeneGroup(const ObjCryst::RefinableObj& obj,
            CrystVector<unsigned int>& groupIndex,
            unsigned int& firstGroup) const
    {
        override f = this->get_override("GetGeneGroup");
        if (f)  f(obj, groupIndex, firstGroup);
        else  default_GetGeneGroup(obj, groupIndex, firstGroup);
    }

    void default_UpdateDisplay() const
    { this->RefinableObj::UpdateDisplay();}

    void UpdateDisplay() const
    {
        override f = this->get_override("UpdateDisplay");
        if (f)  f();
        else  default_UpdateDisplay();
    }

    double default_GetRestraintCost() const
    { return this->RefinableObj::GetRestraintCost();}

    double GetRestraintCost() const
    {
        override f = this->get_override("GetRestraintCost");
        if (f)  return f();
        return default_GetRestraintCost();
    }

    void default_TagNewBestConfig() const
    { this->RefinableObj::TagNewBestConfig();}

    void TagNewBestConfig() const
    {
        override f = this->get_override("TagNewBestConfig");
        if (f)  f();
        else  default_TagNewBestConfig();
    }

    // Protected methods

    void default_Prepare()
    { RefinableObj::Prepare();}

    void Prepare()
    {
        override f = this->get_override("Prepare");
        if (f)  f();
        else  default_Prepare();
    }

    long FindPar(const std::string& name) const
    { return RefinableObj::FindPar(name); }

    void AddOption(RefObjOpt* opt)
    { RefinableObj::AddOption(opt); }

    std::map<unsigned long,
        std::pair<
            CrystVector<double>, std::string>
        >::iterator
        FindParamSet(unsigned long id) const
        { return RefinableObj::FindParamSet(id); }

};

void _RemovePar(RefinableObj& obj, RefinablePar* refpar)
{
    obj.RemovePar(refpar);
    return;
}

void _XMLOutput(
        const RefinableObj& r,
        bp::object output,
        int indent = 0)
{
    boost_adaptbx::python::ostream os(output);
    os.precision(doublelim::digits10);
    r.XMLOutput(os, indent);
    os.flush();
}

void _XMLInput(
        RefinableObj& r,
        bp::object input,
        XMLCrystTag& tag)
{
    boost_adaptbx::python::streambuf sbuf(input);
    boost_adaptbx::python::streambuf::istream in(sbuf);
    r.XMLInput(in, tag);
    in.sync();
}

} // anonymous namespace


void wrap_refinableobj()
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
        .def("GetPar", &_GetParLong,
            return_internal_reference<>())
        .def("GetPar", &_GetParString,
            return_internal_reference<>())
        .def("GetParNotFixed", &_GetParNotFixedLong,
            return_internal_reference<>())
        .def("AddPar", &_AddPar,
            bp::arg("par"),
            with_custodian_and_ward<1,2>())
        .def("AddPar", &_AddParObj,
            (bp::arg("newRefParList"), bp::arg("copyParam")=false),
            with_custodian_and_ward<1,2>())
        .def("RemovePar", &_RemovePar)
        .def("CreateParamSet", &RefinableObj::CreateParamSet,
            (bp::arg("name")=""))
        .def("ClearParamSet", &RefinableObj::ClearParamSet)
        .def("SaveParamSet", &RefinableObj::SaveParamSet)
        .def("RestoreParamSet", &RefinableObj::RestoreParamSet)
        .def("GetParamSet", (const CrystVector<double>& (RefinableObj::*)
            (const unsigned long) const) &RefinableObj::GetParamSet,
            return_value_policy<copy_const_reference>())
        .def("GetParamSet_ParNotFixedHumanValue",
            &RefinableObj::GetParamSet_ParNotFixedHumanValue)
        .def("EraseAllParamSet", &RefinableObjWrap::EraseAllParamSet)
        .def("GetParamSetName", &RefinableObj::GetParamSetName,
            return_value_policy<copy_const_reference>())
        .def("SetLimitsAbsolute", ( void (RefinableObj::*)
            (const std::string&, const double, const double) )
            &RefinableObj::SetLimitsAbsolute)
        .def("SetLimitsAbsolute", ( void (RefinableObj::*)
            (const RefParType*, const double, const double) )
            &RefinableObj::SetLimitsAbsolute)
        .def("SetLimitsRelative", ( void (RefinableObj::*)
            (const std::string&, const double, const double) )
            &RefinableObj::SetLimitsRelative)
        .def("SetLimitsRelative", ( void (RefinableObj::*)
            (const RefParType*, const double, const double) )
            &RefinableObj::SetLimitsRelative)
        .def("SetLimitsProportional", ( void (RefinableObj::*)
            (const std::string&, const double, const double) )
            &RefinableObj::SetLimitsProportional)
        .def("SetLimitsProportional", ( void (RefinableObj::*)
            (const RefParType*, const double, const double) )
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
            with_custodian_and_ward<1,2>())
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
            &RefinableObjWrap::default_BeginOptimization,
            (bp::arg("allowApproximations")=false,
             bp::arg("enableRestraints")=false))
        .def("EndOptimization", &RefinableObj::EndOptimization,
            &RefinableObjWrap::default_EndOptimization)
        .def("RandomizeConfiguration", &RefinableObj::RandomizeConfiguration,
            &RefinableObjWrap::default_RandomizeConfiguration)
        .def("GlobalOptRandomMove", &RefinableObj::GlobalOptRandomMove,
            &RefinableObjWrap::default_GlobalOptRandomMove,
            (bp::arg("mutationAmplitude"), bp::arg("type")=gpRefParTypeObjCryst))
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
        .def("XMLOutput", &_XMLOutput, (bp::arg("file"), bp::arg("indent")=0))
        .def("XMLOutput", &RefinableObj::XMLOutput,
            &RefinableObjWrap::default_XMLOutput)
        .def("XMLInput", &_XMLInput, (bp::arg("file"), bp::arg("tag")))
        .def("XMLInput", &RefinableObj::XMLInput,
            &RefinableObjWrap::default_XMLInput)
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
