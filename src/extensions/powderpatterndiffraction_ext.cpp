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
* boost::python bindings to ObjCryst::PowderPatternDiffraction.
*
* Changes from ObjCryst::PowderPatternDiffraction
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/enum.hpp>
#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>
#undef B0

#include <ObjCryst/ObjCryst/General.h>
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

    class_<PowderPatternDiffraction,
        bases<PowderPatternComponent, ScatteringData> >(
                "PowderPatternDiffraction", no_init)
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
        .def("SetCrystal",
                &PowderPatternDiffraction::SetCrystal,
                bp::arg("crystal"))
        .def("GetNbReflBelowMaxSinThetaOvLambda",
                &PowderPatternDiffraction::GetNbReflBelowMaxSinThetaOvLambda)
        ;
}
