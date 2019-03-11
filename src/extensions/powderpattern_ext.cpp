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
* boost::python bindings to ObjCryst::PowderPattern.
*
* Changes from ObjCryst::PowderPattern
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/manage_new_object.hpp>

#undef B0

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "python_streambuf.hpp"
#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {


PowderPattern* _CreatePowderPatternFromCIF(bp::object input)
{
    // Reading a cif file creates some output. Let's redirect stdout to a junk
    // stream and then throw it away.
    // FIXME ... try to remove this kludge with junk buffer
    ostringstream junk;
    swapstdout(junk);

    boost_adaptbx::python::streambuf sbuf(input);
    boost_adaptbx::python::streambuf::istream in(sbuf);
    ObjCryst::CIF cif(in);

    int idx0 = gPowderPatternRegistry.GetNb();

    ObjCryst::CreatePowderPatternFromCIF(cif);

    int idx = gPowderPatternRegistry.GetNb();

    // Switch the stream buffer back
    swapstdout(junk);

    if(idx == idx0)
    {
        throw ObjCryst::ObjCrystException("Cannot create powder pattern from CIF");
    }
    idx--;

    ObjCryst::PowderPattern* p = &gPowderPatternRegistry.GetObj(idx);

    return p;
}


PowderPatternBackground& addppbackground(PowderPattern& pp)
{
    PowderPatternBackground* ppc = new PowderPatternBackground();
    pp.AddPowderPatternComponent(*ppc);
    return *ppc;
}


PowderPatternDiffraction& addppdiffraction(PowderPattern& pp, Crystal& crst)
{
    PowderPatternDiffraction* ppc = new PowderPatternDiffraction();
    ppc->SetCrystal(crst);
    pp.AddPowderPatternComponent(*ppc);
    return *ppc;
}


void setpowderpatternx (PowderPattern& pp, bp::object x)
{
    CrystVector_REAL cvx;
    assignCrystVector(cvx, x);
    pp.SetPowderPatternX(cvx);
}


void setpowderpatternobs (PowderPattern& pp, bp::object x)
{
    CrystVector_REAL cvx;
    assignCrystVector(cvx, x);
    MuteObjCrystUserInfo muzzle;
    pp.SetPowderPatternObs(cvx);
}


}   // namespace

void wrap_powderpattern()
{
    class_<PowderPattern, bases<RefinableObj> >("PowderPattern")
        .def("AddPowderPatternBackground",
                &addppbackground,
                return_internal_reference<>())
        .def("AddPowderPatternDiffraction",
                &addppdiffraction,
                with_custodian_and_ward<2,1,return_internal_reference<> >())
        .def("GetNbPowderPatternComponent",
                &PowderPattern::GetNbPowderPatternComponent)
        .def("GetPowderPatternComponent",
                (PowderPatternComponent& (PowderPattern::*) (const int))
                &PowderPattern::GetPowderPatternComponent,
                return_internal_reference<>())
        .def("GetScaleFactor",
                (REAL (PowderPattern::*) (const int) const)
                &PowderPattern::GetScaleFactor)
        .def("SetScaleFactor",
                (void (PowderPattern::*) (const int, REAL))
                &PowderPattern::SetScaleFactor)
        .def("SetPowderPatternPar",
                &PowderPattern::SetPowderPatternPar,
                (bp::arg("xmin"), bp::arg("xstep"), bp::arg("nbpoints")))
        .def("SetPowderPatternX",
                &setpowderpatternx,
                bp::arg("x"))
        .def("GetPowderPatternCalc",
                &PowderPattern::GetPowderPatternCalc,
                return_value_policy<copy_const_reference>())
        .def("GetPowderPatternObs",
                &PowderPattern::GetPowderPatternObs,
                return_value_policy<copy_const_reference>())
        .def("GetPowderPatternX",
                &PowderPattern::GetPowderPatternX,
                return_value_policy<copy_const_reference>())
        .def("SetWavelength",
                (void (PowderPattern::*) (const REAL))
                &PowderPattern::SetWavelength, bp::arg("wavelength"))
        .def("SetWavelength",
                (void (PowderPattern::*) (const string&, const REAL))
                &PowderPattern::SetWavelength,
                (bp::arg("XRayTubeElementName"), bp::arg("alpha2Alpha2ratio")=0.5))
        .def("SetEnergy",
                &DiffractionDataSingleCrystal::SetEnergy,
                bp::arg("nrj_kev"))
        .def("ImportPowderPatternFullprof",
                &PowderPattern::ImportPowderPatternFullprof,
                bp::arg("filename"))
        .def("ImportPowderPatternPSI_DMC",
                &PowderPattern::ImportPowderPatternPSI_DMC,
                bp::arg("filename"))
        .def("ImportPowderPatternILL_D1A5",
                &PowderPattern::ImportPowderPatternILL_D1A5,
                bp::arg("filename"))
        .def("ImportPowderPatternXdd",
                &PowderPattern::ImportPowderPatternXdd,
                bp::arg("filename"))
        .def("ImportPowderPatternSietronicsCPI",
                &PowderPattern::ImportPowderPatternSietronicsCPI,
                bp::arg("filename"))
        .def("ImportPowderPattern2ThetaObsSigma",
                &PowderPattern::ImportPowderPattern2ThetaObsSigma,
                (bp::arg("filename"), bp::arg("nbSkip")=0))
        .def("ImportPowderPatternFullprof4",
                &PowderPattern::ImportPowderPatternFullprof4,
                bp::arg("filename"))
        .def("ImportPowderPatternMultiDetectorLLBG42",
                &PowderPattern::ImportPowderPatternMultiDetectorLLBG42,
                bp::arg("filename"))
        .def("ImportPowderPattern2ThetaObs",
                &PowderPattern::ImportPowderPattern2ThetaObs,
                (bp::arg("filename"), bp::arg("nbSkip")=0))
        .def("ImportPowderPatternTOF_ISIS_XYSigma",
                &PowderPattern::ImportPowderPatternTOF_ISIS_XYSigma,
                bp::arg("filename"))
        .def("ImportPowderPatternGSAS",
                &PowderPattern::ImportPowderPatternGSAS,
                bp::arg("filename"))
        .def("SetPowderPatternObs",
                &setpowderpatternobs,
                bp::arg("obs"))
        .def("FitScaleFactorForR",
                &PowderPattern::FitScaleFactorForR)
        .def("FitScaleFactorForIntegratedR",
                &PowderPattern::FitScaleFactorForIntegratedR)
        .def("FitScaleFactorForRw",
                &PowderPattern::FitScaleFactorForRw)
        .def("FitScaleFactorForIntegratedRw",
                &PowderPattern::FitScaleFactorForIntegratedRw)
        .def("SetMaxSinThetaOvLambda",
                &PowderPattern::SetMaxSinThetaOvLambda,
                bp::arg("max"))
        .def("GetMaxSinThetaOvLambda",
                &PowderPattern::GetMaxSinThetaOvLambda)
        .def("Prepare", &PowderPattern::Prepare)
        ;

    def("CreatePowderPatternFromCIF",
            &_CreatePowderPatternFromCIF, bp::arg("file"),
            return_value_policy<manage_new_object>());
}
