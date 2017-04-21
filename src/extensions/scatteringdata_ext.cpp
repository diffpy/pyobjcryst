/*****************************************************************************
*
* pyobjcryst        Complex Modeling Initiative
*                   (c) 2015 Brookhaven Science Associates
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Kevin Knox
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::ScatteringData.  These bindings are
* used by ObjCryst objects that inherit from ScatteringData (see, for example,
* diffractiondatasinglecrystal_ext.cpp).
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/CrystVector/CrystVector.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

void _PrintFhklCalc(const ScatteringData& sd)
{
    sd.PrintFhklCalc();
}


void _PrintFhklCalcDetail(const ScatteringData& sd)
{
    sd.PrintFhklCalcDetail();
}

}   // anonymous namespace


void wrap_scatteringdata()
{

    class_<ScatteringData, bases<RefinableObj>, boost::noncopyable>(
            "ScatteringData", no_init)
        /* Methods */
        // Have to convert from Python array to C++ array
        //.def("SetHKL", &ScatteringData::SetHKL)
        .def("GenHKLFullSpace2",
                &ScatteringData::GenHKLFullSpace2,
                (bp::arg("maxsithsl"), bp::arg("unique")=false))
        .def("GenHKLFullSpace",
                &ScatteringData::GenHKLFullSpace,
                (bp::arg("maxtheta"), bp::arg("unique")=false))
        // have to figure this out
        //.def("GetRadiationType", &ScatteringData::GetRadiationType,
        //    return_value_policy<copy_const_reference>())
        .def("SetCrystal", &ScatteringData::SetCrystal)
        .def("GetCrystal",
                (Crystal& (ScatteringData::*)()) &ScatteringData::GetCrystal,
                return_internal_reference<>())
        .def("HasCrystal", &ScatteringData::HasCrystal)
        .def("GetNbRefl", &ScatteringData::GetNbRefl)
        .def("GetH", &ScatteringData::GetH,
                return_value_policy<copy_const_reference>())
        .def("GetK", &ScatteringData::GetK,
                return_value_policy<copy_const_reference>())
        .def("GetL", &ScatteringData::GetL,
                return_value_policy<copy_const_reference>())
        .def("GetH2Pi", &ScatteringData::GetH2Pi,
                return_value_policy<copy_const_reference>())
        .def("GetK2Pi", &ScatteringData::GetK2Pi,
                return_value_policy<copy_const_reference>())
        .def("GetL2Pi", &ScatteringData::GetL2Pi,
                return_value_policy<copy_const_reference>())
        .def("GetReflX", &ScatteringData::GetReflX,
                return_value_policy<copy_const_reference>())
        .def("GetReflY", &ScatteringData::GetReflY,
                return_value_policy<copy_const_reference>())
        .def("GetReflZ", &ScatteringData::GetReflZ,
                return_value_policy<copy_const_reference>())
        .def("GetSinThetaOverLambda", &ScatteringData::GetSinThetaOverLambda,
                return_value_policy<copy_const_reference>())
        .def("GetTheta", &ScatteringData::GetTheta,
                return_value_policy<copy_const_reference>())
        .def("GetClockTheta", &ScatteringData::GetClockTheta,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcSq", &ScatteringData::GetFhklCalcSq,
                return_value_policy<copy_const_reference>())
        //.def("GetFhklCalcSq_FullDeriv")
        .def("GetFhklCalcReal", &ScatteringData::GetFhklCalcReal,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcImag", &ScatteringData::GetFhklCalcImag,
                return_value_policy<copy_const_reference>())
        .def("GetFhklObsSq", &ScatteringData::GetFhklObsSq,
                return_value_policy<copy_const_reference>())
        //.def("GetScatteringFactor",
        //     (const std::map< const ScatteringPower *, CrystVector_REAL > &
        //     (ScatteringData::*)()) &ScatteringData::GetScatteringFactor,
        //     return_internal_reference<>())
        .def("GetWavelength", &ScatteringData::GetWavelength)
        .def("SetIsIgnoringImagScattFact",
                &ScatteringData::SetIsIgnoringImagScattFact)
        .def("IsIgnoringImagScattFact",
                &ScatteringData::IsIgnoringImagScattFact)
        .def("PrintFhklCalc", &_PrintFhklCalc)
        .def("PrintFhklCalcDetail", &_PrintFhklCalcDetail)
        // These don't seem necessary as I doubt we'll use ObjCryst for optimizations
        //.def("BeginOptimization")
        //.def("EndOptimization")
        //.def("SetApproximationFlag")
        .def("SetMaxSinThetaOvLambda", &ScatteringData::SetMaxSinThetaOvLambda)
        .def("GetMaxSinThetaOvLambda", &ScatteringData::GetMaxSinThetaOvLambda)
        .def("GetNbReflBelowMaxSinThetaOvLambda",
                &ScatteringData::GetNbReflBelowMaxSinThetaOvLambda)
        .def("GetClockNbReflBelowMaxSinThetaOvLambda",
                &ScatteringData::GetClockNbReflBelowMaxSinThetaOvLambda,
                return_value_policy<copy_const_reference>())
        ;
}
