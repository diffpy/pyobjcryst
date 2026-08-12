/*
 * pyobjcryst nanobind port — RefinablePar bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// Wrapper to store the double value alongside the parameter
class PyRefinablePar : public RefinablePar
{
public:
    PyRefinablePar() : RefinablePar(), pval(nullptr) {}

    PyRefinablePar(const std::string& name, double value,
                   const double min, const double max,
                   const RefParType* type,
                   RefParDerivStepModel derivMode = REFPAR_DERIV_STEP_RELATIVE,
                   const bool hasLimits = true,
                   const bool isFixed = false,
                   const bool isUsed = true,
                   const bool isPeriodic = false,
                   const REAL humanScale = 1.,
                   REAL period = 1.)
        : RefinablePar()
    {
        pval = new REAL(static_cast<REAL>(value));
        RefinablePar::Init(name, pval, min, max, type, derivMode,
                           hasLimits, isFixed, isUsed, isPeriodic, humanScale, period);
    }

    ~PyRefinablePar()
    {
        delete pval;
    }

private:
    REAL* pval;
};

} // namespace

void wrap_refinablepar(nb::module_& m)
{
    nb::enum_<RefParDerivStepModel>(m, "RefParDerivStepModel")
        .value("REFPAR_DERIV_STEP_ABSOLUTE", REFPAR_DERIV_STEP_ABSOLUTE)
        .value("REFPAR_DERIV_STEP_RELATIVE", REFPAR_DERIV_STEP_RELATIVE)
        .export_values()
        ;

    // C++-created RefinablePar (no init from python)
    nb::class_<RefinablePar, Restraint>(m, "_RefinablePar")
        .def("GetValue",   &RefinablePar::GetValue)
        .def("SetValue",   &RefinablePar::SetValue)
        .def("GetHumanValue", &RefinablePar::GetHumanValue)
        .def("SetHumanValue", &RefinablePar::SetHumanValue)
        .def("Mutate",     &RefinablePar::Mutate)
        .def("MutateTo",   &RefinablePar::MutateTo)
        .def("GetSigma",   &RefinablePar::GetSigma)
        .def("GetHumanSigma", &RefinablePar::GetHumanSigma)
        .def("SetSigma",   &RefinablePar::SetSigma)
        .def("GetName",    &RefinablePar::GetName)
        .def("SetName",    &RefinablePar::SetName)
        .def("Print",      &RefinablePar::Print)
        .def("IsFixed",    &RefinablePar::IsFixed)
        .def("SetIsFixed", &RefinablePar::SetIsFixed)
        .def("IsLimited",  &RefinablePar::IsLimited)
        .def("SetIsLimited",&RefinablePar::SetIsLimited)
        .def("IsUsed",     &RefinablePar::IsUsed)
        .def("SetIsUsed",  &RefinablePar::SetIsUsed)
        .def("IsPeriodic", &RefinablePar::IsPeriodic)
        .def("SetIsPeriodic",&RefinablePar::SetIsPeriodic)
        .def("GetHumanScale",&RefinablePar::GetHumanScale)
        .def("SetHumanScale",&RefinablePar::SetHumanScale)
        .def("GetMin",     &RefinablePar::GetMin)
        .def("SetMin",     &RefinablePar::SetMin)
        .def("GetHumanMin",&RefinablePar::GetHumanMin)
        .def("SetHumanMin",&RefinablePar::SetHumanMin)
        .def("GetMax",     &RefinablePar::GetMax)
        .def("SetMax",     &RefinablePar::SetMax)
        .def("GetHumanMax",&RefinablePar::GetHumanMax)
        .def("SetHumanMax",&RefinablePar::SetHumanMax)
        .def("GetPeriod",  &RefinablePar::GetPeriod)
        .def("SetPeriod",  &RefinablePar::SetPeriod)
        .def("GetDerivStep",&RefinablePar::GetDerivStep)
        .def("SetDerivStep",&RefinablePar::SetDerivStep)
        .def("GetGlobalOptimStep",&RefinablePar::GetGlobalOptimStep)
        .def("SetGlobalOptimStep",&RefinablePar::SetGlobalOptimStep)
        .def("AssignClock",&RefinablePar::AssignClock)
        .def("SetLimitsAbsolute",&RefinablePar::SetLimitsAbsolute)
        .def("SetLimitsRelative",&RefinablePar::SetLimitsRelative)
        .def("SetLimitsProportional",&RefinablePar::SetLimitsProportional)
        .def("GetType",    &RefinablePar::GetType, nb::rv_policy::reference_internal)
        .def("__str__",    [](const RefinablePar& p){ return obj_str(p); })
        .def_prop_rw("value", &RefinablePar::GetValue, &RefinablePar::SetValue)
        ;

    // Python-created RefinablePar
    nb::class_<PyRefinablePar, RefinablePar>(m, "RefinablePar")
        .def(nb::init<const std::string&, double, const REAL, const double,
                      const RefParType*, RefParDerivStepModel,
                      const bool, const bool, const bool, const bool,
                      const double, double>(),
             nb::arg("name"), nb::arg("value"), nb::arg("min"), nb::arg("max"),
             nb::arg("type"),
             nb::arg("derivMode") = REFPAR_DERIV_STEP_RELATIVE,
             nb::arg("hasLimits") = true,
             nb::arg("isFixed") = false,
             nb::arg("isUsed") = true,
             nb::arg("isPeriodic") = false,
             nb::arg("humanScale") = 1.,
             nb::arg("period") = 1.,
             nb::keep_alive<1, 6>())
        ;
}
