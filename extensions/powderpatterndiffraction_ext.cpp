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
* boost::python bindings to ObjCryst::PowderPattern.
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
#include <ObjCryst/ObjCryst/PowderPattern.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


void wrap_powderpatterndiffraction()
{
    enum_<ReflectionProfileType>("ReflectionProfileType")
        .value("PROFILE_GAUSSIAN", PROFILE_GAUSSIAN)
        .value("PROFILE_LORENTZIAN", PROFILE_LORENTZIAN)
        .value("PROFILE_PSEUDO_VOIGT", PROFILE_PSEUDO_VOIGT)
        .value("PROFILE_PSEUDO_VOIGT_FINGER_COX_JEPHCOAT",
                PROFILE_PSEUDO_VOIGT_FINGER_COX_JEPHCOAT)
        .value("PROFILE_PEARSON_VII", PROFILE_PEARSON_VII)
        ;

    class_<PowderPatternDiffraction, bases<PowderPatternComponent> >(
            "PowderPatternDiffraction")
        .def("GetPowderPatternCalc",
                &PowderPatternDiffraction::GetPowderPatternCalc,
                return_value_policy<copy_const_reference>())
        .def("SetReflectionProfilePar",
                &PowderPatternDiffraction::SetReflectionProfilePar,
                (bp::arg("type")=PROFILE_PSEUDO_VOIGT,
                 bp::arg("fwhmCagliotiW")=1e-6,
                 bp::arg("fwhmCagliotiU")=0,
                 bp::arg("fwhmCagliotiV")=0,
                 bp::arg("eta0")=0.5, bp::arg("eta1")=0))
        .def("GetProfile",
                (ReflectionProfile& (PowderPatternDiffraction::*)())
                &PowderPatternDiffraction::GetProfile,
                return_internal_reference<>())
        .def("SetExtractionMode",
                &PowderPatternDiffraction::SetExtractionMode,
                (bp::arg("extract")=true, bp::arg("init")=false))
        .def("GetExtractionMode",
                &PowderPatternDiffraction::GetExtractionMode)
        .def("ExtractLeBail",
                &PowderPatternDiffraction::ExtractLeBail,
                (bp::arg("nbcycle")=1))

        //.def("Prepare", &PowderPatternDiffraction::Prepare) // protected

        // From ScatteringData:
        .def("SetCrystal",
                &PowderPatternDiffraction::SetCrystal,
                bp::arg("crystal"))
        .def("GetNbRefl",
                &ScatteringData::GetNbRefl)
        .def("GetH",
                &ScatteringData::GetH,
                return_value_policy<copy_const_reference>())
        .def("GetK",
                &ScatteringData::GetK,
                return_value_policy<copy_const_reference>())
        .def("GetL",
                &ScatteringData::GetL,
                return_value_policy<copy_const_reference>())
        .def("GetReflX",
                &ScatteringData::GetReflX,
                return_value_policy<copy_const_reference>())
        .def("GetReflY",
                &ScatteringData::GetReflY,
                return_value_policy<copy_const_reference>())
        .def("GetReflZ",
                &ScatteringData::GetReflZ,
                return_value_policy<copy_const_reference>())
        .def("GetSinThetaOverLambda",
                &ScatteringData::GetSinThetaOverLambda,
                return_value_policy<copy_const_reference>())
        .def("GetTheta",
                &ScatteringData::GetTheta,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcSq",
                &ScatteringData::GetFhklCalcSq,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcReal",
                &ScatteringData::GetFhklCalcReal,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcImag",
                &ScatteringData::GetFhklCalcImag,
                return_value_policy<copy_const_reference>())
        .def("GetFhklObsSq",
                &ScatteringData::GetFhklObsSq,
                return_value_policy<copy_const_reference>())
        ;
}
