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
* boost::python bindings to ObjCryst::LSQNumObj.
*
* Changes from ObjCryst::LSQNumObj
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/RefinableObj/LSQNumObj.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

bool _SafeRefine(LSQNumObj & lsq, REAL maxChi2factor, int nbCycle, bool useLevenbergMarquardt,
                 const bool silent, const bool callBeginEndOptimization,const float minChi2var)
{
    CaptureStdOut gag;
    if(!silent) gag.release();

    std::list<RefinablePar*> vnewpar;
    std::list<const RefParType*> vnewpartype;
    return lsq.SafeRefine(vnewpar, vnewpartype, nbCycle, useLevenbergMarquardt, silent,
                          callBeginEndOptimization, minChi2var);
}

}   // namespace

void wrap_lsq()
{
    class_<LSQNumObj>("LSQ")
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
        .def("SafeRefine", &_SafeRefine,
                (bp::arg("maxChi2factor")=1.01,bp::arg("nbCycle")=1,
                 bp::arg("useLevenbergMarquardt")=false,
                 bp::arg("silent")=false,
                 bp::arg("callBeginEndOptimization")=true,
                 bp::arg("minChi2var")=0.01))
        .def("Rfactor", &LSQNumObj::Rfactor)
        .def("RwFactor", &LSQNumObj::RwFactor)
        .def("ChiSquare", &LSQNumObj::ChiSquare)
        .def("SetRefinedObj", &LSQNumObj::SetRefinedObj,
                (bp::arg("obj"), bp::arg("LSQFuncIndex")=0,
                 bp::arg("init")=true, bp::arg("recursive")=false),
                 with_custodian_and_ward<1,2>())
        .def("GetCompiledRefinedObj",
                (RefinableObj& (LSQNumObj::*)())
                &LSQNumObj::GetCompiledRefinedObj,
                with_custodian_and_ward_postcall<0,1,return_internal_reference<> >())
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
                bp::arg("par"),
                return_value_policy<copy_const_reference>())
        .def("BeginOptimization", &LSQNumObj::BeginOptimization,
                (bp::arg("allowApproximations")=false,
                 bp::arg("enableRestraints")=false))
        .def("EndOptimization", &LSQNumObj::EndOptimization)
        ;
}
