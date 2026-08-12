/*
 * pyobjcryst nanobind port — LSQNumObj bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/RefinableObj/LSQNumObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

bool _SafeRefine(LSQNumObj& lsq, REAL maxChi2factor, int nbCycle,
                 bool useLevenbergMarquardt, const bool silent,
                 const bool callBeginEndOptimization, const float minChi2var)
{
    CaptureStdOut gag;
    if (!silent) gag.release();
    std::list<RefinablePar*> vnewpar;
    std::list<const RefParType*> vnewpartype;
    return lsq.SafeRefine(vnewpar, vnewpartype, nbCycle, useLevenbergMarquardt,
                          silent, callBeginEndOptimization, minChi2var);
}

// LSQNumObj::SetRefinedObj(RefinableObj&, ...) generically accepts "any
// RefinableObj" -- see the extractRefinableObjArg comment in helpers_nb.hpp
// for why a plain `&LSQNumObj::SetRefinedObj` binding fails for
// virtually-linked concrete types (e.g. DiffractionDataSingleCrystal).
void _SetRefinedObj(LSQNumObj& lsq, nb::object obj,
                    const unsigned int LSQFuncIndex, const bool init,
                    const bool recursive)
{
    lsq.SetRefinedObj(extractRefinableObjArg(obj), LSQFuncIndex, init, recursive);
}

} // namespace

void wrap_lsq(nb::module_& m)
{
    nb::class_<LSQNumObj>(m, "LSQ")
        .def(nb::init<>())
        .def("SetParIsFixed",
             nb::overload_cast<const std::string&, const bool>(&LSQNumObj::SetParIsFixed),
             nb::arg("parName"), nb::arg("fix"))
        .def("SetParIsFixed",
             nb::overload_cast<const RefParType*, const bool>(&LSQNumObj::SetParIsFixed),
             nb::arg("type"), nb::arg("fix"))
        .def("SetParIsFixed",
             nb::overload_cast<RefinablePar&, const bool>(&LSQNumObj::SetParIsFixed),
             nb::arg("par"), nb::arg("fix"))
        .def("UnFixAllPar", &LSQNumObj::UnFixAllPar)
        .def("Refine", &LSQNumObj::Refine,
             nb::arg("nbCycle") = 1,
             nb::arg("useLevenbergMarquardt") = false,
             nb::arg("silent") = false,
             nb::arg("callBeginEndOptimization") = true,
             nb::arg("minChi2var") = 0.01f)
        .def("SafeRefine", &_SafeRefine,
             nb::arg("maxChi2factor") = 1.01,
             nb::arg("nbCycle") = 1,
             nb::arg("useLevenbergMarquardt") = false,
             nb::arg("silent") = false,
             nb::arg("callBeginEndOptimization") = true,
             nb::arg("minChi2var") = 0.01f)
        .def("Rfactor",   &LSQNumObj::Rfactor)
        .def("RwFactor",  &LSQNumObj::RwFactor)
        .def("ChiSquare", &LSQNumObj::ChiSquare)
        .def("SetRefinedObj", &_SetRefinedObj,
             nb::arg("obj"), nb::arg("LSQFuncIndex") = 0,
             nb::arg("init") = true, nb::arg("recursive") = false,
             nb::keep_alive<1,2>())
        .def("GetCompiledRefinedObj",
             nb::overload_cast<>(&LSQNumObj::GetCompiledRefinedObj),
             nb::rv_policy::reference_internal)
        .def("PrintRefResults",  &LSQNumObj::PrintRefResults)
        .def("PrepareRefParList",&LSQNumObj::PrepareRefParList,
             nb::arg("copy_param") = false)
        .def("GetLSQCalc",
             [](LSQNumObj& obj){ return crystvec_to_array(obj.GetLSQCalc()); })
        .def("GetLSQObs",
             [](LSQNumObj& obj){ return crystvec_to_array(obj.GetLSQObs()); })
        .def("GetLSQWeight",
             [](LSQNumObj& obj){ return crystvec_to_array(obj.GetLSQWeight()); })
        .def("GetLSQDeriv",
             [](LSQNumObj& obj, RefinablePar& par){ return crystvec_to_array(obj.GetLSQDeriv(par)); },
             nb::arg("par"))
        .def("BeginOptimization", &LSQNumObj::BeginOptimization,
             nb::arg("allowApproximations") = false,
             nb::arg("enableRestraints") = false)
        .def("EndOptimization", &LSQNumObj::EndOptimization)
        ;
}
