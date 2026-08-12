/*
 * pyobjcryst nanobind port — RefinableObj bindings
 *
 * Changes from ObjCryst::RefinableObj:
 * - XMLOutput and XMLInput accept python file-like objects or strings
 * - SetDeleteRefParInDestructor(false) called in constructors
 * - GetParamSet returns a copy
 * - RemovePar returns None
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <sstream>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>

// Extra headers needed only for extractRefinableObjArg()'s fallback chain
// (see helpers_nb.hpp) -- every concrete class that virtually inherits
// RefinableObj and is therefore unreachable from it via nanobind's own
// mechanism.
#undef B0
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/Scatterer.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

// ---------------------------------------------------------------------------
// Trampoline for RefinableObj virtual methods
// ---------------------------------------------------------------------------
struct PyRefinableObj : RefinableObj {
    NB_TRAMPOLINE(RefinableObj, 14);

    // Note: GetClassName() and GetName() return const string& which can't be overridden
    // via NB_OVERRIDE (returns temporary). Subclasses should use SetName instead.
    void SetName(const std::string& n) override { NB_OVERRIDE(SetName, n); }
    void Print() const override { NB_OVERRIDE(Print); }
    void RegisterClient(RefinableObj& c) const override { NB_OVERRIDE(RegisterClient, c); }
    void DeRegisterClient(RefinableObj& c) const override { NB_OVERRIDE(DeRegisterClient, c); }
    ObjRegistry<RefinableObj>& GetClientRegistry() override { NB_OVERRIDE(GetClientRegistry); }
    void BeginOptimization(const bool approx, const bool restr) override {
        NB_OVERRIDE(BeginOptimization, approx, restr);
    }
    void EndOptimization() override { NB_OVERRIDE(EndOptimization); }
    void RandomizeConfiguration() override { NB_OVERRIDE(RandomizeConfiguration); }
    void GlobalOptRandomMove(const REAL amp, const RefParType* type) override {
        NB_OVERRIDE(GlobalOptRandomMove, amp, type);
    }
    REAL GetLogLikelihood() const override { NB_OVERRIDE(GetLogLikelihood); }
    unsigned int GetNbLSQFunction() const override { NB_OVERRIDE(GetNbLSQFunction); }
    void UpdateDisplay() const override { NB_OVERRIDE(UpdateDisplay); }
    REAL GetRestraintCost() const override { NB_OVERRIDE(GetRestraintCost); }
};

// ---------------------------------------------------------------------------
// Helper wrappers
// ---------------------------------------------------------------------------
namespace {

void _AddPar(RefinableObj& obj, RefinablePar* p)
{
    obj.AddPar(p);
    obj.SetDeleteRefParInDestructor(0);
}

void _AddParObj(RefinableObj& obj, nb::object o, const bool copyParam = false)
{
    obj.AddPar(extractRefinableObjArg(o), copyParam);
    obj.SetDeleteRefParInDestructor(0);
}

RefinablePar& _GetParLong(RefinableObj& obj, const long i)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetPar(i);
}

RefinablePar& _GetParString(RefinableObj& obj, const std::string& s)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetPar(s);
}

RefinablePar& _GetParNotFixed(RefinableObj& obj, const long i)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetParNotFixed(i);
}

void _RemovePar(RefinableObj& obj, RefinablePar* p) { obj.RemovePar(p); }

std::string _xml(const RefinableObj& r)
{
    std::stringstream s;
    r.XMLOutput(s, 0);
    return s.str();
}

void _XMLOutput(const RefinableObj& r, nb::object output, int indent = 0)
{
    std::ostringstream os;
    os.precision(doublelim::digits10);
    r.XMLOutput(os, indent);
    output.attr("write")(os.str());
}

void _XMLInput(RefinableObj& r, nb::object input, XMLCrystTag& tag)
{
    std::string s = read_pyfile_to_string(input);
    std::istringstream is(s);
    r.XMLInput(is, tag);
}

void _XMLInputString(RefinableObj& r, const std::string& s)
{
    std::istringstream ss(s);
    ss.imbue(std::locale::classic());
    XMLCrystTag tag;
    ss >> tag;
    r.XMLInput(ss, tag);
}

// Return CrystVector as a numpy array copy
nb_array_1d _GetParamSet(const RefinableObj& obj, const unsigned long id)
{
    return crystvec_to_array(obj.GetParamSet(id));
}

} // namespace

// External-linkage helper declared in helpers_nb.hpp -- see the comment
// there for why this is needed and why it lives here.
RefinableObj& extractRefinableObjArg(nb::object obj)
{
    // Safe/declared path first: works for anything reaching RefinableObj
    // through a non-virtual chain (Crystal, UnitCell, Molecule, Atom,
    // PowderPattern, MonteCarloObj, LSQNumObj, ...).
    try { return nb::cast<RefinableObj&>(obj); } catch (...) {}
    // Known classes that virtually inherit RefinableObj (directly or via an
    // intermediate virtual base) and are therefore registered in nanobind
    // without a declared path to it -- try each concrete/intermediate type,
    // then bridge with a real (always-correct) static_cast.
    try { return static_cast<RefinableObj&>(nb::cast<ScatteringData&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<PowderPatternComponent&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<Scatterer&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<ScatteringPower&>(obj)); } catch (...) {}
    try { return static_cast<RefinableObj&>(nb::cast<ScatteringPowerAtom&>(obj)); } catch (...) {}
    // Nothing matched -- let this raise a clear nanobind TypeError rather
    // than silently returning a dangling reference.
    return nb::cast<RefinableObj&>(obj);
}

void wrap_refinableobj(nb::module_& m)
{
    m.attr("refpartype_objcryst") = gpRefParTypeObjCryst;
    m.attr("gRefinableObjRegistry") = &gRefinableObjRegistry;
    m.attr("gTopRefinableObjRegistry") = &gTopRefinableObjRegistry;

    nb::class_<RefinableObj, PyRefinableObj>(m, "RefinableObj")
        .def(nb::init<>())
        .def(nb::init<bool>())
        /* Parameter management */
        .def("PrepareForRefinement", &RefinableObj::PrepareForRefinement)
        .def("FixAllPar",   &RefinableObj::FixAllPar)
        .def("UnFixAllPar", &RefinableObj::UnFixAllPar)
        .def("SetParIsFixed",
             nb::overload_cast<const long, const bool>(&RefinableObj::SetParIsFixed))
        .def("SetParIsFixed",
             nb::overload_cast<const std::string&, const bool>(&RefinableObj::SetParIsFixed))
        .def("SetParIsFixed",
             nb::overload_cast<const RefParType*, const bool>(&RefinableObj::SetParIsFixed))
        .def("SetParIsUsed",
             nb::overload_cast<const std::string&, const bool>(&RefinableObj::SetParIsUsed))
        .def("SetParIsUsed",
             nb::overload_cast<const RefParType*, const bool>(&RefinableObj::SetParIsUsed))
        .def("GetNbPar",          &RefinableObj::GetNbPar)
        .def("GetNbParNotFixed",  &RefinableObj::GetNbParNotFixed)
        .def("GetPar",
             &_GetParLong,    nb::rv_policy::reference_internal)
        .def("GetPar",
             &_GetParString,  nb::rv_policy::reference_internal)
        .def("GetParNotFixed", &_GetParNotFixed, nb::rv_policy::reference_internal)
        .def("AddPar", &_AddPar,    nb::arg("par"), nb::keep_alive<1,2>())
        .def("AddPar", &_AddParObj, nb::arg("newRefParList"), nb::arg("copyParam") = false,
             nb::keep_alive<1,2>())
        .def("RemovePar", &_RemovePar)
        /* Parameter sets */
        .def("CreateParamSet", &RefinableObj::CreateParamSet,
             nb::arg("name") = "")
        .def("ClearParamSet",  &RefinableObj::ClearParamSet)
        .def("SaveParamSet",   &RefinableObj::SaveParamSet)
        .def("RestoreParamSet",&RefinableObj::RestoreParamSet)
        .def("GetParamSet",    &_GetParamSet)
        .def("GetParamSet_ParNotFixedHumanValue",
             &RefinableObj::GetParamSet_ParNotFixedHumanValue)
        .def("EraseAllParamSet", [](RefinableObj& obj){ obj.EraseAllParamSet(); })
        .def("GetParamSetName",&RefinableObj::GetParamSetName)
        /* Limits */
        .def("SetLimitsAbsolute",
             nb::overload_cast<const std::string&, const REAL, const REAL>(
                 &RefinableObj::SetLimitsAbsolute))
        .def("SetLimitsAbsolute",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(
                 &RefinableObj::SetLimitsAbsolute))
        .def("SetLimitsRelative",
             nb::overload_cast<const std::string&, const REAL, const REAL>(
                 &RefinableObj::SetLimitsRelative))
        .def("SetLimitsRelative",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(
                 &RefinableObj::SetLimitsRelative))
        .def("SetLimitsProportional",
             nb::overload_cast<const std::string&, const REAL, const REAL>(
                 &RefinableObj::SetLimitsProportional))
        .def("SetLimitsProportional",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(
                 &RefinableObj::SetLimitsProportional))
        .def("SetGlobalOptimStep", &RefinableObj::SetGlobalOptimStep)
        /* Registries / sub-objects */
        .def("GetSubObjRegistry",
             nb::overload_cast<>(&RefinableObj::GetSubObjRegistry),
             nb::rv_policy::reference_internal)
        .def("GetClientRegistry",
             nb::overload_cast<>(&RefinableObj::GetClientRegistry),
             nb::rv_policy::reference_internal)
        .def("GetOptionList", &RefinableObj::GetOptionList,
             nb::rv_policy::reference)
        .def("GetNbOption",   &RefinableObj::GetNbOption)
        .def("GetOption",
             nb::overload_cast<const unsigned int>(&RefinableObj::GetOption),
             nb::rv_policy::reference_internal)
        /* State */
        .def("IsBeingRefined",          &RefinableObj::IsBeingRefined)
        .def("BeginGlobalOptRandomMove",&RefinableObj::BeginGlobalOptRandomMove)
        .def("ResetParList",            &RefinableObj::ResetParList)
        .def("GetRefParListClock",      &RefinableObj::GetRefParListClock,
             nb::rv_policy::reference_internal)
        .def("GetClockMaster",          &RefinableObj::GetClockMaster,
             nb::rv_policy::reference_internal)
        .def("AddRestraint",   &RefinableObj::AddRestraint,  nb::keep_alive<1,2>())
        .def("RemoveRestraint",&RefinableObj::RemoveRestraint)
        /* Virtual methods */
        .def("GetClassName",  &RefinableObj::GetClassName)
        .def("GetName",       &RefinableObj::GetName)
        .def("SetName",       &RefinableObj::SetName)
        .def("Print",         &RefinableObj::Print)
        .def("RegisterClient",  &RefinableObj::RegisterClient,
             nb::keep_alive<1,2>())
        .def("DeRegisterClient",&RefinableObj::DeRegisterClient)
        .def("BeginOptimization",&RefinableObj::BeginOptimization,
             nb::arg("allowApproximations") = false,
             nb::arg("enableRestraints") = false)
        .def("EndOptimization",       &RefinableObj::EndOptimization)
        .def("RandomizeConfiguration",&RefinableObj::RandomizeConfiguration)
        .def("GlobalOptRandomMove",   &RefinableObj::GlobalOptRandomMove,
             nb::arg("mutationAmplitude"),
             nb::arg("type") = gpRefParTypeObjCryst)
        .def("GetLogLikelihood",  &RefinableObj::GetLogLikelihood)
        .def("GetNbLSQFunction",  &RefinableObj::GetNbLSQFunction)
        // LSQ-target extensibility API -- lets a Python subclass supply a
        // custom least-squares objective (data/weights/derivatives). Ported
        // forward from Boost.Python (missing from the initial nanobind
        // port); returns numpy arrays like every other CrystVector-returning
        // method here.
        .def("GetLSQCalc",   [](RefinableObj& o, unsigned int i){
                                 return crystvec_to_array(o.GetLSQCalc(i)); },
             nb::arg("idx"))
        .def("GetLSQObs",    [](RefinableObj& o, unsigned int i){
                                 return crystvec_to_array(o.GetLSQObs(i)); },
             nb::arg("idx"))
        .def("GetLSQWeight", [](RefinableObj& o, unsigned int i){
                                 return crystvec_to_array(o.GetLSQWeight(i)); },
             nb::arg("idx"))
        .def("GetLSQDeriv",  [](RefinableObj& o, unsigned int i, RefinablePar& p){
                                 return crystvec_to_array(o.GetLSQDeriv(i, p)); },
             nb::arg("idx"), nb::arg("par"))
        .def("UpdateDisplay",     &RefinableObj::UpdateDisplay)
        .def("GetRestraintCost",  &RefinableObj::GetRestraintCost)
        .def("TagNewBestConfig",  &RefinableObj::TagNewBestConfig)
        /* XML I/O */
        .def("XMLOutput",  &_XMLOutput,   nb::arg("file"), nb::arg("indent") = 0)
        .def("xml",        &_xml)
        .def("XMLInput",   &_XMLInput,    nb::arg("file"), nb::arg("tag"))
        .def("XMLInput",   &_XMLInputString, nb::arg("xml"))
        .def("GetGeneGroup",&RefinableObj::GetGeneGroup)
        .def("int_ptr",    &RefinableObj::int_ptr)
        .def("__str__",    [](const RefinableObj& obj){ return obj_str(obj); })
        ;
}
