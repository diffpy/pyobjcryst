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
* boost::python bindings to ObjCryst::LSQNumObj.
*
* Changes from ObjCryst::LSQNumObj
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

#include <numpy/noprefix.h>
#include <numpy/arrayobject.h>

#include <string>
#include <map>

#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>
#include <ObjCryst/RefinableObj/LSQNumObj.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

}   // namespace

void wrap_lsqnumobj()
{
    class_<LSQNumObj>("LSQNumObj", init<>())
        /// LSQNumObj::PrepareRefParList() must be called first!
        .def("SetParIsFixed",
                (void (LSQNumObj::*)(const std::string&, const bool))
                &LSQNumObj::SetParIsFixed,
                (bp::arg("parName"), bp::arg("fix")))
        .def("SetParIsFixed",
                (void (LSQNumObj::*)(const RefParType *, const bool))
                &LSQNumObj::SetParIsFixed,
                (bp::arg("type"), bp::arg("fix")))
        .def("SetParIsFixed",
                (void (LSQNumObj::*)(RefinablePar &, const bool))
                &LSQNumObj::SetParIsFixed,
                (bp::arg("par"), bp::arg("fix")))
        //void SetParIsFixed(RefinableObj &obj, const bool fix);
        .def("UnFixAllPar", &LSQNumObj::UnFixAllPar)
        //void SetParIsUsed(const std::string& parName, const bool use);
        //void SetParIsUsed(const RefParType *type, const bool use);
        .def("Refine", &LSQNumObj::Refine,
                (bp::arg("nbCycle")=1,
                 bp::arg("useLevenbergMarquardt")=false,
                 bp::arg("silent")=false,
                 bp::arg("callBeginEndOptimization")=true,
                 bp::arg("minChi2var")=0.01))
        .def("Rfactor", &LSQNumObj::Rfactor)
        .def("RwFactor", &LSQNumObj::RwFactor)
        .def("ChiSquare", &LSQNumObj::ChiSquare)
        .def("SetRefinedObj", &LSQNumObj::SetRefinedObj,
                (bp::arg("obj"), bp::arg("LSQFuncIndex")=0,
                 bp::arg("init")=true, bp::arg("recursive")=false))
        .def("GetCompiledRefinedObj",
                (RefinableObj& (LSQNumObj::*)())
                &LSQNumObj::GetCompiledRefinedObj,
                return_internal_reference<>())
        .def("PrintRefResults", &LSQNumObj::PrintRefResults)
        .def("PrepareRefParList", &LSQNumObj::PrepareRefParList,
                (bp::arg("copy_param")=false))
        .def("GetLSQCalc", &LSQNumObj::GetLSQCalc,
                return_value_policy<copy_const_reference>())
        .def("GetLSQObs", &LSQNumObj::GetLSQObs,
                return_value_policy<copy_const_reference>())
        .def("GetLSQWeight", &LSQNumObj::GetLSQWeight,
                return_value_policy<copy_const_reference>())
        .def("GetLSQDeriv", &LSQNumObj::GetLSQDeriv,
                (bp::arg("par")),
                return_value_policy<copy_const_reference>())
        .def("BeginOptimization", &LSQNumObj::BeginOptimization,
                (bp::arg("allowApproximations")=false,
                 bp::arg("enableRestraints")=false))
        .def("EndOptimization", &LSQNumObj::EndOptimization)
        ;
}
