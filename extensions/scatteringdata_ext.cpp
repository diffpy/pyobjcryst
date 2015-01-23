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
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::ScatteringData. This is a virtual class
* that can be derived from in python. These bindings are used by ObjCryst
* objects that inherit from ScatteringData (see, for example,
* diffractiondatasinglecrystal_ext.cpp).  ScatteringData derivatives can be
* created in python and will work in c++ functions that are also bound into
* python.
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <string>
#include <map>
#include <iostream>

#include <ObjCryst/ObjCryst/PDF.h>
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/RefinableObj/IO.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers.hpp"
#include "python_file_stream.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {


class ScatteringDataWrap : public ScatteringData,
                 public wrapper<ScatteringData>
{

    public:
    
    void default_SetHKL(const CrystVector_REAL &h, const CrystVector_REAL &k,
			const CrystVector_REAL &l)
    { this->ScatteringData::SetHKL(h, k, l); }

    void SetHKL(const CrystVector_REAL &h, const CrystVector_REAL &k,
		const CrystVector_REAL &l)
    {
	if (override SetHKL = this->get_override("SetHKL"))
	    SetHKL(h, k, l);
	default_SetHKL(h, k, l);
    }

    void default_GenHKLFullSpace2(const double maxsithsl, const bool unique=false)
    { this->ScatteringData::GenHKLFullSpace2(maxsithsl, unique); }

    void GenHKLFullSpace2(const double maxsithsl, const bool unique=false)
    {
	if (override GenHKLFullSpace2 = this->get_override("GenHKLFullSpace2"))
	    GenHKLFullSpace2(maxsithsl, unique);
	default_GenHKLFullSpace2(maxsithsl, unique);
    }

    void default_GenHKLFullSpace(const double maxsithsl, const bool unique=false)
    { this->ScatteringData::GenHKLFullSpace(maxsithsl, unique); }

    void GenHKLFullSpace(const double maxsithsl, const bool unique=false)
    {
	if (override GenHKLFullSpace = this->get_override("GenHKLFullSpace"))
	    GenHKLFullSpace(maxsithsl, unique);
	default_GenHKLFullSpace(maxsithsl, unique);
    }

    void default_SetCrystal(Crystal &crystal)
    { this->ScatteringData::SetCrystal(crystal); }

    void SetCrystal(Crystal &crystal)
    {
	if (override SetCrystal = this->get_override("SetCrystal"))
	    SetCrystal(crystal);
	default_SetCrystal(crystal);
    }

    void default_SetMaxSinThetaOvLambda(const double max)
    { this->ScatteringData::SetMaxSinThetaOvLambda(max); }

    void SetMaxSinThetaOvLambda(const double max)
    {
	if (override SetMaxSinThetaOvLambda = 
	    this->get_override("SetMaxSinThetaOvLambda"))
	    SetMaxSinThetaOvLambda(max);
	default_SetMaxSinThetaOvLambda(max);
    }

    long default_GetNbReflBelowMaxSinThetaOvLambda() const
    { this->ScatteringData::GetNbReflBelowMaxSinThetaOvLambda(); }

    long GetNbReflBelowMaxSinThetaOvLambda() const
    {
	if (override GetNbReflBelowMaxSinThetaOvLambda = 
	    this->get_override("GetNbReflBelowMaxSinThetaOvLambda"))
	    GetNbReflBelowMaxSinThetaOvLambda();
	default_GetNbReflBelowMaxSinThetaOvLambda();
    }

};

void _PrintFhklCalc(const ScatteringData& sd)
{
    sd.PrintFhklCalc();
}	    


void _PrintFhklCalcDetail(const ScatteringData& sd)
{
    sd.PrintFhklCalcDetail();
}	    

} //anonymous namespace

void wrap_scatteringdata()
{

    class_<ScatteringDataWrap, bases<RefinableObj>,
	boost::noncopyable>("ScatteringData", no_init)
        /* Methods */
	// Have to convert from Python array to C++ array
	//.def("SetHKL", &ScatteringData::SetHKL,
	//   &ScatteringDataWrap::default_SetHKL)
        .def("GenHKLFullSpace2", &ScatteringData::GenHKLFullSpace2,
            &ScatteringDataWrap::default_GenHKLFullSpace2, bp::arg("unique")=false)
        .def("GenHKLFullSpace", &ScatteringData::GenHKLFullSpace,
            &ScatteringDataWrap::default_GenHKLFullSpace, bp::arg("unique")=false)
	// have to figure this out
	//.def("GetRadiationType", &ScatteringData::GetRadiationType,
	//    return_value_policy<copy_const_reference>())
        .def("SetCrystal", &ScatteringData::SetCrystal,
            &ScatteringDataWrap::default_SetCrystal)
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
	 //.def("GetClockTheta")
	.def("GetFhklCalcSq", &ScatteringData::GetFhklCalcSq,
	     return_value_policy<copy_const_reference>())
	//.def("GetFhklCalcSq_FullDeriv")
	.def("GetFhklCalcReal", &ScatteringData::GetFhklCalcReal,
	     return_value_policy<copy_const_reference>())
	.def("GetFhklCalcImag", &ScatteringData::GetFhklCalcImag,
	     return_value_policy<copy_const_reference>())
	//.def("GetFhklObsSq")
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
	.def("SetMaxSinThetaOvLambda", &ScatteringData::SetMaxSinThetaOvLambda,
	     &ScatteringDataWrap::default_SetMaxSinThetaOvLambda)
	.def("GetMaxSinThetaOvLambda", &ScatteringData::GetMaxSinThetaOvLambda)
	.def("GetNbReflBelowMaxSinThetaOvLambda",
	     &ScatteringData::GetNbReflBelowMaxSinThetaOvLambda,
	     &ScatteringDataWrap::default_GetNbReflBelowMaxSinThetaOvLambda)
	//.def("GetClockNbReflBelowMaxSinThetaOvLambda")
	;
}


