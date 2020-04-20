/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::GlobalOptimObj and ObjCryst::MonteCarloObj.
*
* Changes from ObjCryst::MonteCarloObj:
* - add access to mAutoLSQ option
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/enum.hpp>

#include <ObjCryst/RefinableObj/GlobalOptimObj.h>
#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overload MonteCarlo class so that it does not auto-save to XML
class MonteCarloObjWrap: public MonteCarloObj
{
  public:
    MonteCarloObjWrap(const string name=""):
    MonteCarloObj(name)
    {
      this->GetOption("Save Best Config Regularly").SetChoice(0);
    }
};

void run_optimize(MonteCarloObjWrap& obj, long nbSteps, const bool silent,
        const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.Optimize(nbSteps, silent, finalcost, maxTime);
}

void multirun_optimize(MonteCarloObjWrap& obj, long nbCycle, long nbSteps,
        const bool silent, const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.MultiRunOptimize(nbCycle, nbSteps, silent, finalcost, maxTime);
}

void mc_sa(MonteCarloObjWrap& obj, long nbSteps, const bool silent,
        const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.RunSimulatedAnnealing(nbSteps, silent, finalcost, maxTime);
}

void mc_pt(MonteCarloObjWrap& obj, long nbSteps, const bool silent,
        const double finalcost, const double maxTime)
{
    CaptureStdOut gag;
    obj.RunParallelTempering(nbSteps, silent, finalcost, maxTime);
}

/*
void mc_random_lsq(MonteCarloObjWrap& obj, long nbCycle)
{
    obj.RunRandomLSQMethod(nbCycle);
}
*/

}   // namespace


void wrap_globaloptim()
{
    enum_<AnnealingSchedule>("AnnealingSchedule")
        .value("CONSTANT", ANNEALING_CONSTANT)
        .value("BOLTZMANN", ANNEALING_BOLTZMANN)
        .value("CAUCHY", ANNEALING_CAUCHY)
        .value("EXPONENTIAL", ANNEALING_EXPONENTIAL)
        .value("SMART", ANNEALING_SMART)
        .value("GAMMA", ANNEALING_GAMMA)
        ;

    enum_<GlobalOptimType>("GlobalOptimType")
        .value("SIMULATED_ANNEALING", GLOBAL_OPTIM_SIMULATED_ANNEALING)
        .value("PARALLEL_TEMPERING", GLOBAL_OPTIM_PARALLEL_TEMPERING)
        .value("RANDOM_LSQ", GLOBAL_OPTIM_RANDOM_LSQ)
        .value("SIMULATED_ANNEALING_MULTI", GLOBAL_OPTIM_SIMULATED_ANNEALING_MULTI)
        .value("PARALLEL_TEMPERING_MULTI", GLOBAL_OPTIM_PARALLEL_TEMPERING_MULTI)
        ;

    class_<MonteCarloObjWrap>("MonteCarlo")
        //////////////// OptimizationObj methods
        .def("RandomizeStartingConfig", &MonteCarloObj::RandomizeStartingConfig)
        .def("Optimize", &run_optimize,
                (bp::arg("nbSteps"), bp::arg("silent")=false,
                 bp::arg("finalcost")=0.0, bp::arg("maxTime")=-1))
        .def("MultiRunOptimize", &multirun_optimize,
                (bp::arg("nbCycle"), bp::arg("nbSteps"), bp::arg("silent")=false,
                 bp::arg("finalcost")=0.0, bp::arg("maxTime")=-1))
        .def("FixAllPar", &MonteCarloObj::FixAllPar)
        .def("SetParIsFixed", (void (MonteCarloObj::*)(const std::string&, const bool))
            &MonteCarloObj::SetParIsFixed)
        .def("SetParIsFixed", (void (MonteCarloObj::*)(const RefParType*, const bool))
            &MonteCarloObj::SetParIsFixed)
        .def("UnFixAllPar", &MonteCarloObj::UnFixAllPar)
        .def("SetParIsUsed", (void (MonteCarloObj::*)(const std::string&, const bool))
            &MonteCarloObj::SetParIsUsed)
        .def("SetParIsUsed", (void (MonteCarloObj::*)(const RefParType*, const bool))
            &MonteCarloObj::SetParIsUsed)
        .def("SetLimitsRelative", ( void (MonteCarloObj::*)
            (const std::string&, const double, const double) )
            &MonteCarloObj::SetLimitsRelative)
        .def("SetLimitsRelative", ( void (MonteCarloObj::*)
            (const RefParType*, const double, const double) )
            &MonteCarloObj::SetLimitsRelative)
        .def("SetLimitsAbsolute", ( void (MonteCarloObj::*)
            (const std::string&, const double, const double) )
            &MonteCarloObj::SetLimitsAbsolute)
        .def("SetLimitsAbsolute", ( void (MonteCarloObj::*)
            (const RefParType*, const double, const double) )
            &MonteCarloObj::SetLimitsAbsolute)
        .def("GetLogLikelihood", &MonteCarloObj::GetLogLikelihood)
        .def("StopAfterCycle", &MonteCarloObj::StopAfterCycle)
        .def("AddRefinableObj", &MonteCarloObj::AddRefinableObj,
            bp::arg("obj"),
            with_custodian_and_ward<1,2>())
        .def("GetFullRefinableObj", &MonteCarloObj::GetFullRefinableObj,
            bp::arg("rebuild")=true,
            with_custodian_and_ward_postcall<1,0, return_internal_reference<> >())
        .def("GetName", &MonteCarloObj::GetName,
            return_value_policy<copy_const_reference>())
        .def("SetName", &MonteCarloObj::SetName)
        .def("Print", &MonteCarloObj::Print, &MonteCarloObj::Print)
        .def("RestoreBestConfiguration", &MonteCarloObj::RestoreBestConfiguration)
        .def("IsOptimizing", &MonteCarloObj::IsOptimizing)
        .def("GetLastOptimElapsedTime", &MonteCarloObj::GetLastOptimElapsedTime)
        //.def("GetMainTracker", &MonteCarloObj::GetMainTracker)
        //.add_property("name", &MonteCarloObj::GetName, &MonteCarloObj::SetName)
        .def("GetNbOption", &MonteCarloObj::GetNbOption)
        .def("GetOption", (RefObjOpt& (MonteCarloObj::*)(const unsigned int))
            &MonteCarloObj::GetOption,
            return_internal_reference<>())
        .def("GetOption", (RefObjOpt& (MonteCarloObj::*)(const string&))
            &MonteCarloObj::GetOption,
            with_custodian_and_ward_postcall<1,0,return_internal_reference<> >())

        //////////////// MonteCarlo methods
        .def("SetAlgorithmParallTempering", &MonteCarloObj::SetAlgorithmParallTempering,
             (bp::arg("scheduleTemp"), bp::arg("tMax"), bp::arg("tMin"),
              bp::arg("scheduleMutation")=ANNEALING_SMART,
              bp::arg("mutMax")=16, bp::arg("mutMin")=.125))
        .def("SetAlgorithmSimulAnnealing", &MonteCarloObj::SetAlgorithmSimulAnnealing,
             (bp::arg("scheduleTemp"), bp::arg("tMax"), bp::arg("tMin"),
              bp::arg("scheduleMutation")=ANNEALING_SMART,
              bp::arg("mutMax")=16, bp::arg("mutMin")=.125,
              bp::arg("nbTrialRetry")=0, bp::arg("minCostRetry")=0.))
        .def("RunSimulatedAnnealing", &mc_sa,
             (bp::arg("nbSteps"), bp::arg("silent")=false,
              bp::arg("finalcost")=0.0, bp::arg("maxTime")=-1))
        .def("RunParallelTempering", &mc_pt,
             (bp::arg("nbSteps"), bp::arg("silent")=false,
              bp::arg("finalcost")=0.0, bp::arg("maxTime")=-1))
        // TODO: seems unstable
        //.def("RunRandomLSQ", &mc_random_lsq,
        //        bp::arg("nbCycle"))
        .def("GetLSQObj",
             (LSQNumObj& (MonteCarloObj::*)()) &MonteCarloObj::GetLSQObj,
             with_custodian_and_ward_postcall<1,0, return_internal_reference<> >())
        .def("InitLSQ", &MonteCarloObj::InitLSQ,
             bp::arg("useFullPowderPatternProfile")=true)
        ;
}
