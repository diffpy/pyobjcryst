/*
 * pyobjcryst nanobind port — PowderPatternComponent bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#undef B0
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

// PowderPatternComponent : virtual public RefinableObj (see PowderPattern.h)
// -- registered standalone (no declared nanobind base), per the virtual
// inheritance pattern documented in nanobind_migration_notes.md §1. Base
// RefinableObj methods needed by callers (e.g. PowderPatternBackground's
// UnFixAllPar(), used from the Python-layer quick_fit_profile()) must be
// rebound explicitly here via static_cast, not inherited. Any class that
// safely nanobind-inherits PowderPatternComponent (i.e. via a non-virtual
// link -- see PowderPatternBackground) picks these up automatically;
// PowderPatternDiffraction inherits PowderPatternComponent virtually and
// therefore repeats the same rebinding itself (see nb_powderpatterndiffraction.cpp).
void wrap_powderpatterncomponent(nb::module_& m)
{
    nb::class_<PowderPatternComponent> cls(m, "PowderPatternComponent");
    cls
        .def("GetParentPowderPattern",
             nb::overload_cast<>(&PowderPatternComponent::GetParentPowderPattern),
             nb::rv_policy::reference_internal)
        .def("GetName", [](PowderPatternComponent& c) -> std::string {
                            return static_cast<RefinableObj&>(c).GetName(); })
        .def("SetName", [](PowderPatternComponent& c, const std::string& n) {
                            static_cast<RefinableObj&>(c).SetName(n); })
        .def("GetClassName", [](PowderPatternComponent& c) -> std::string {
                            return static_cast<RefinableObj&>(c).GetClassName(); })
        .def("FixAllPar",   [](PowderPatternComponent& c) { static_cast<RefinableObj&>(c).FixAllPar(); })
        .def("UnFixAllPar", [](PowderPatternComponent& c) { static_cast<RefinableObj&>(c).UnFixAllPar(); })
        ;

    // Restore the rest of RefinableObj's method surface (XMLOutput,
    // BeginOptimization, GetLogLikelihood, AddPar, CreateParamSet,
    // GetOption, UpdateDisplay, ... -- only GetName/SetName/GetClassName/
    // FixAllPar/UnFixAllPar were covered one-off above). See helpers_nb.hpp
    // for why this is safe despite PowderPatternComponent's virtual
    // inheritance of RefinableObj. Covers PowderPatternBackground
    // automatically (non-virtual inheritance, already correctly declared as
    // its nanobind base). PowderPatternDiffraction inherits
    // PowderPatternComponent virtually too, but already gets this same
    // surface via its (non-virtual) ScatteringData base once
    // nb_scatteringdata.cpp's copy is applied, so it needs no separate call.
    bind_refinableobj_forwarding(cls);
}
