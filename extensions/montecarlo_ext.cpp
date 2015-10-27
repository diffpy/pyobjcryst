/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::MonteCarloObj.
*
* Changes from ObjCryst::MonteCarloObj
*
* Other Changes
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/slice.hpp>

#include <string>
#include <map>

#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>
#include <ObjCryst/RefinableObj/GlobalOptimObj.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

void mcoptimize(MonteCarloObj &obj, long nbSteps, const bool silent,
        const double finalcost, const double maxTime)
{
    obj.Optimize(nbSteps, silent, finalcost, maxTime);
}

}   // namespace


void wrap_montecarloobj()
{
    class_<MonteCarloObj>("MonteCarloObj") //, bases<OptimizationObj>
        //        .def("GetBestCost", &MonteCarloObj::GetBestCost)
        .def("IsOptimizing", &MonteCarloObj::IsOptimizing)
        //        .add_property("Name", &MonteCarloObj::GetName, &MonteCarloObj::SetName)
        .def("RandomizeStartingConfig", &MonteCarloObj::RandomizeStartingConfig)
        .def("Optimize", &mcoptimize,
                (bp::arg("nbSteps"), bp::arg("silent")=false,
                 bp::arg("finalcost")=0.0, bp::arg("maxTime")=-1))
        //.def("RunParallelTempering", &MonteCarloObj::RunParallelTempering,
        //	(bp::arg("nbSteps"), bp::arg("silent"), bp::arg("finalcost"), bp::arg("maxTime")))
        // From OptimizationObj:
        .def("AddRefinableObj", &MonteCarloObj::AddRefinableObj)
        .def("GetLogLikelihood", &MonteCarloObj::GetLogLikelihood)
        ;
}
