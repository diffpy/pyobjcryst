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
#include <ObjCryst/ObjCryst/ReflectionProfile.h>
#include <ObjCryst/ObjCryst/ScatteringCorr.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>

#include <string>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace
{

void SetProfileCopy(PowderPatternDiffraction &pdiff,
                    const ReflectionProfile &profile)
{
    pdiff.SetProfile(profile.CreateCopy());
}

// PowderPatternDiffraction has no public accessor for its texture
// correction, but registers it as a sub-object.
TextureMarchDollase &GetTextureMarchDollase(PowderPatternDiffraction &pdiff)
{
    ObjRegistry<RefinableObj> &subobjs = pdiff.GetSubObjRegistry();
    for (long i = 0; i < subobjs.GetNb(); ++i)
    {
        TextureMarchDollase *texture =
            dynamic_cast<TextureMarchDollase *>(&subobjs.GetObj(i));
        if (texture != 0) return *texture;
    }
    throw ObjCrystException("PowderPatternDiffraction has no "
                            "TextureMarchDollase sub-object. Please report "
                            "this as a pyobjcryst bug.");
}

// TextureMarchDollase::SetPhasePar does not click the texture clock, so the
// cached powder pattern is not recalculated. Set the values through the
// phase's RefinablePars instead, which AddPhase registers in this order.
void SetPhasePar(TextureMarchDollase &texture, const unsigned int i,
                 const REAL fraction, const REAL coeffMarch,
                 const REAL h, const REAL k, const REAL l)
{
    if (i >= texture.GetNbPhase())
    {
        throw ObjCrystException("TextureMarchDollase has no phase " +
                                std::to_string(i) + ". Please use an index "
                                "smaller than GetNbPhase().");
    }
    const REAL values[5] = {fraction, coeffMarch, h, k, l};
    for (long j = 0; j < 5; ++j)
        texture.GetPar(5L * i + j).SetValue(values[j]);
}

} // namespace


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
        .def("SetProfile", &SetProfileCopy, bp::arg("profile"),
                "Install an independent copy of a reflection profile.")
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
                bp::arg("crystal"),
                with_custodian_and_ward<1, 2>())
        .def("GetNbReflBelowMaxSinThetaOvLambda",
                &PowderPatternDiffraction::GetNbReflBelowMaxSinThetaOvLambda)
        .def("GetFhklObsSq", &PowderPatternDiffraction::GetFhklObsSq,
                return_value_policy<copy_const_reference>())
        .def("X2XCorrPhase", &PowderPatternDiffraction::X2XCorrPhase)
        .def("GetTextureMarchDollase", &GetTextureMarchDollase,
                return_internal_reference<>(),
                "Return the March-Dollase preferred-orientation correction.")
        ;

    class_<TextureMarchDollase, bases<RefinableObj>, boost::noncopyable>(
            "TextureMarchDollase", no_init)
        .def("AddPhase", &TextureMarchDollase::AddPhase,
                (bp::arg("fraction"), bp::arg("coeffMarch"), bp::arg("h"),
                 bp::arg("k"), bp::arg("l")))
        .def("SetPhasePar", &SetPhasePar,
                (bp::arg("i"), bp::arg("fraction"), bp::arg("coeffMarch"),
                 bp::arg("h"), bp::arg("k"), bp::arg("l")))
        .def("GetNbPhase", &TextureMarchDollase::GetNbPhase)
        .def("GetFraction", &TextureMarchDollase::GetFraction, bp::arg("i"))
        .def("GetMarchCoeff", &TextureMarchDollase::GetMarchCoeff,
                bp::arg("i"))
        .def("GetPhaseH", &TextureMarchDollase::GetPhaseH, bp::arg("i"))
        .def("GetPhaseK", &TextureMarchDollase::GetPhaseK, bp::arg("i"))
        .def("GetPhaseL", &TextureMarchDollase::GetPhaseL, bp::arg("i"))
        ;

}
