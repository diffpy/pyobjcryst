/*
 * pyobjcryst nanobind port — GlobalOptimObj / MonteCarloObj bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/GlobalOptimObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyMonteCarloObj : MonteCarloObj {
    NB_TRAMPOLINE(MonteCarloObj, 1);
    void UpdateDisplay() const override { NB_OVERRIDE(UpdateDisplay); MonteCarloObj::UpdateDisplay(); }
};

namespace {

void run_optimize(MonteCarloObj& obj, long nbSteps, const bool silent,
                  const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.Optimize(nbSteps, silent, finalcost, maxTime);
}

// MonteCarloObj::AddRefinableObj(RefinableObj&) generically accepts "any
// RefinableObj" -- see the extractRefinableObjArg comment in helpers_nb.hpp
// for why a plain `&MonteCarloObj::AddRefinableObj` binding fails for
// virtually-linked concrete types (e.g. DiffractionDataSingleCrystal).
void _AddRefinableObj(MonteCarloObj& mc, nb::object obj)
{
    mc.AddRefinableObj(extractRefinableObjArg(obj));
}

void multirun_optimize(MonteCarloObj& obj, long nbCycle, long nbSteps,
                       const bool silent, const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.MultiRunOptimize(nbCycle, nbSteps, silent, finalcost, maxTime);
}

void mc_sa(MonteCarloObj& obj, long nbSteps, const bool silent,
           const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.RunSimulatedAnnealing(nbSteps, silent, finalcost, maxTime);
}

void mc_pt(MonteCarloObj& obj, long nbSteps, const bool silent,
           const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.RunParallelTempering(nbSteps, silent, finalcost, maxTime);
}

} // namespace

void wrap_globaloptim(nb::module_& m)
{
    m.attr("gOptimizationObjRegistry") = &gOptimizationObjRegistry;

    nb::enum_<AnnealingSchedule>(m, "AnnealingSchedule")
        .value("CONSTANT",    ANNEALING_CONSTANT)
        .value("BOLTZMANN",   ANNEALING_BOLTZMANN)
        .value("CAUCHY",      ANNEALING_CAUCHY)
        .value("EXPONENTIAL", ANNEALING_EXPONENTIAL)
        .value("SMART",       ANNEALING_SMART)
        .value("GAMMA",       ANNEALING_GAMMA)
        .export_values()
        ;

    nb::enum_<GlobalOptimType>(m, "GlobalOptimType")
        .value("SIMULATED_ANNEALING",         GLOBAL_OPTIM_SIMULATED_ANNEALING)
        .value("PARALLEL_TEMPERING",          GLOBAL_OPTIM_PARALLEL_TEMPERING)
        .value("RANDOM_LSQ",                  GLOBAL_OPTIM_RANDOM_LSQ)
        .value("SIMULATED_ANNEALING_MULTI",   GLOBAL_OPTIM_SIMULATED_ANNEALING_MULTI)
        .value("PARALLEL_TEMPERING_MULTI",    GLOBAL_OPTIM_PARALLEL_TEMPERING_MULTI)
        .export_values()
        ;

    nb::class_<MonteCarloObj, PyMonteCarloObj>(m, "MonteCarlo")
        .def(nb::init<>())
        .def(nb::init<const std::string&>(), nb::arg("name"))
        .def("RandomizeStartingConfig", &MonteCarloObj::RandomizeStartingConfig)
        .def("Optimize", &run_optimize,
             nb::arg("nbSteps"), nb::arg("silent") = false,
             nb::arg("finalcost") = 0.0, nb::arg("maxTime") = -1.0)
        .def("MultiRunOptimize", &multirun_optimize,
             nb::arg("nbCycle"), nb::arg("nbSteps"), nb::arg("silent") = false,
             nb::arg("finalcost") = 0.0, nb::arg("maxTime") = -1.0)
        .def("FixAllPar",   &MonteCarloObj::FixAllPar)
        .def("SetParIsFixed",
             nb::overload_cast<const std::string&, const bool>(&MonteCarloObj::SetParIsFixed))
        .def("SetParIsFixed",
             nb::overload_cast<const RefParType*, const bool>(&MonteCarloObj::SetParIsFixed))
        .def("UnFixAllPar", &MonteCarloObj::UnFixAllPar)
        .def("SetParIsUsed",
             nb::overload_cast<const std::string&, const bool>(&MonteCarloObj::SetParIsUsed))
        .def("SetParIsUsed",
             nb::overload_cast<const RefParType*, const bool>(&MonteCarloObj::SetParIsUsed))
        .def("SetLimitsRelative",
             nb::overload_cast<const std::string&, const REAL, const REAL>(
                 &MonteCarloObj::SetLimitsRelative))
        .def("SetLimitsRelative",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(
                 &MonteCarloObj::SetLimitsRelative))
        .def("SetLimitsAbsolute",
             nb::overload_cast<const std::string&, const REAL, const REAL>(
                 &MonteCarloObj::SetLimitsAbsolute))
        .def("SetLimitsAbsolute",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(
                 &MonteCarloObj::SetLimitsAbsolute))
        .def("GetLogLikelihood",  &MonteCarloObj::GetLogLikelihood)
        .def("StopAfterCycle",    &MonteCarloObj::StopAfterCycle)
        .def("AddRefinableObj",   &_AddRefinableObj,
             nb::arg("obj"), nb::keep_alive<1,2>())
        .def("GetFullRefinableObj", &MonteCarloObj::GetFullRefinableObj,
             nb::arg("rebuild") = true, nb::rv_policy::reference_internal)
        .def("GetName",      &MonteCarloObj::GetName)
        .def("SetName",      &MonteCarloObj::SetName)
        .def("GetClassName", &MonteCarloObj::GetClassName)
        .def("Print",        &MonteCarloObj::Print)
        .def("RestoreBestConfiguration", &MonteCarloObj::RestoreBestConfiguration)
        .def("GetNbParamSet",     &MonteCarloObj::GetNbParamSet)
        .def("GetParamSetIndex",  &MonteCarloObj::GetParamSetIndex)
        .def("GetParamSetCost",   &MonteCarloObj::GetParamSetCost)
        .def("RestoreParamSet",   &MonteCarloObj::RestoreParamSet,
             nb::arg("idx"), nb::arg("update_display") = true)
        .def("IsOptimizing",      &MonteCarloObj::IsOptimizing)
        .def("GetLastOptimElapsedTime", &MonteCarloObj::GetLastOptimElapsedTime)
        .def("GetNbOption",  &MonteCarloObj::GetNbOption)
        .def("GetOption",
             nb::overload_cast<const unsigned int>(&MonteCarloObj::GetOption),
             nb::rv_policy::reference_internal)
        .def("GetOption",
             nb::overload_cast<const std::string&>(&MonteCarloObj::GetOption),
             nb::rv_policy::reference_internal)
        .def_prop_ro("trial", &MonteCarloObj::GetTrial)
        .def_prop_ro("run",   &MonteCarloObj::GetRun)
        .def_prop_ro("llk",   &MonteCarloObj::GetLogLikelihood)
        .def("SetAlgorithmParallTempering",
             &MonteCarloObj::SetAlgorithmParallTempering,
             nb::arg("scheduleTemp"), nb::arg("tMax"), nb::arg("tMin"),
             nb::arg("scheduleMutation") = ANNEALING_SMART,
             nb::arg("mutMax") = 16.0, nb::arg("mutMin") = 0.125)
        .def("SetAlgorithmSimulAnnealing",
             &MonteCarloObj::SetAlgorithmSimulAnnealing,
             nb::arg("scheduleTemp"), nb::arg("tMax"), nb::arg("tMin"),
             nb::arg("scheduleMutation") = ANNEALING_SMART,
             nb::arg("mutMax") = 16.0, nb::arg("mutMin") = 0.125,
             nb::arg("nbTrialRetry") = 0, nb::arg("minCostRetry") = 0.)
        .def("RunSimulatedAnnealing", &mc_sa,
             nb::arg("nbSteps"), nb::arg("silent") = false,
             nb::arg("finalcost") = 0.0, nb::arg("maxTime") = -1.0)
        .def("RunParallelTempering", &mc_pt,
             nb::arg("nbSteps"), nb::arg("silent") = false,
             nb::arg("finalcost") = 0.0, nb::arg("maxTime") = -1.0)
        .def("GetLSQObj",
             nb::overload_cast<>(&MonteCarloObj::GetLSQObj),
             nb::rv_policy::reference_internal)
        .def("InitLSQ", &MonteCarloObj::InitLSQ,
             nb::arg("useFullPowderPatternProfile") = true)
        ;
}
