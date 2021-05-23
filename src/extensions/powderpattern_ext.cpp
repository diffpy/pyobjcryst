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
#include <boost/format.hpp>
#undef B0

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/CIF.h>

#include "python_streambuf.hpp"
#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {


// This creates a C++ PowderPattern object
PowderPattern* _CreatePowderPatternFromCIF(bp::object input)
{
    // Reading a cif file creates some output via fpObjCrystInformUser.
    // Mute the output and restore it on return or exception.
    // Also mute any hardcoded output to cout.
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;

    boost_adaptbx::python::streambuf sbuf(input);
    boost_adaptbx::python::streambuf::istream in(sbuf);
    ObjCryst::CIF cif(in);

    int idx0 = gPowderPatternRegistry.GetNb();

    ObjCryst::PowderPattern* p = ObjCryst::CreatePowderPatternFromCIF(cif);

    gag.release();
    muzzle.release();

    int idx = gPowderPatternRegistry.GetNb();
    if(idx == idx0)
    {
        throw ObjCryst::ObjCrystException("Cannot create powder pattern from CIF");
    }

    return p;
}

// This reads the CIF into an existing PowderPattern object
PowderPattern* _CreatePowderPatternFromCIF(bp::object input, PowderPattern &p)
{
    // Reading a cif file creates some output via fpObjCrystInformUser.
    // Mute the output and restore it on return or exception.
    // Also mute any hardcoded output to cout.
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;

    boost_adaptbx::python::streambuf sbuf(input);
    boost_adaptbx::python::streambuf::istream in(sbuf);
    ObjCryst::CIF cif(in);

    bool import_ok = false;

    for(map<string,CIFData>::iterator pos=cif.mvData.begin();pos!=cif.mvData.end();++pos)
    {
       if(pos->second.mPowderPatternObs.size()>10)
       {
          p.ImportPowderPatternCIF(cif);
          (*fpObjCrystInformUser)((boost::format("CIF: Imported POWDER PATTERN, with %d points") % p.GetNbPoint()).str());
          import_ok = true;
          break; // only import one powder pattern
       }
    }

    gag.release();
    muzzle.release();

    if(!import_ok)
    {
        throw ObjCryst::ObjCrystException("Cannot create powder pattern from CIF");
    }

    return &p;
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
    pp.Prepare();
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


// Allow override (since we can't benefit from override in RefinableObjWrap)
class PowderPatternWrap : public PowderPattern, public wrapper<PowderPattern>
{
  public:
    PowderPatternWrap() : PowderPattern() {}

    PowderPatternWrap(const PowderPattern& p) : PowderPattern(p){}

    PeakList _FindPeaks(const float dmin=2.0,const float maxratio=0.01,
                       const unsigned int maxpeak=100, const bool verbose=true)
    {
      CaptureStdOut gag;
      if(verbose) gag.release();
      return this->FindPeaks(dmin, maxratio, maxpeak);
    }

    void default_UpdateDisplay() const
    { this->PowderPattern::UpdateDisplay();}

    virtual void UpdateDisplay() const
    {
        override f = this->get_override("UpdateDisplay");
        if (f)  f();
        else  default_UpdateDisplay();
    }

};

std::string __str__SPGScore(SPGScore& s)
{
    if(s.ngof>0.0001)
      return (boost::format("%-13s nGoF=%9.4f GoF=%8.3f Rw=%5.2f [%3d reflections, extinct446=%3d]")
                           % s.hm % s.ngof % s.gof % s.rw %s.nbreflused %s.nbextinct446).str();

    return (boost::format("%-13s GoF=%8.3f Rw=%4.2f [extinct446=%3d]")
                          % s.hm % s.gof % s.rw %s.nbreflused %s.nbextinct446).str();
}

bp::list _GetScores(const SpaceGroupExplorer &spgex)
{
    return containerToPyList(spgex.GetScores());
}

}   // namespace

void wrap_powderpattern()
{
    // Global object registry
    scope().attr("gPowderPatternRegistry") = boost::cref(gPowderPatternRegistry);

    class_<PowderPatternWrap, bases<RefinableObj> >("PowderPattern")
        .def("AddPowderPatternBackground",
                &addppbackground,
                return_internal_reference<>())
        .def("AddPowderPatternDiffraction",
                &addppdiffraction,
                with_custodian_and_ward_postcall<1,0,
                  with_custodian_and_ward_postcall<0,2,return_internal_reference<> > > ())
        .def("GetNbPowderPatternComponent",
                &PowderPattern::GetNbPowderPatternComponent)
        .def("GetPowderPatternComponent",
                (PowderPatternComponent& (PowderPattern::*) (const int))
                &PowderPattern::GetPowderPatternComponent,
                return_internal_reference<>())
        .def("FindPeaks", &PowderPatternWrap::_FindPeaks,
                (bp::arg("dmin")=2.0, bp::arg("maxratio")=0.01,
                 bp::arg("maxpeak")=100, bp::arg("verbose")=false))
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
        .def("GetRadiation",
                (Radiation& (PowderPattern::*)()) &PowderPattern::GetRadiation,
                return_internal_reference<>())
        .def("GetRadiationType", &PowderPattern::GetRadiationType)
        .def("SetRadiationType", &PowderPattern::SetRadiationType)
        .def("GetWavelength", &PowderPattern::GetWavelength)
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
        .def("X2XCorr", &PowderPattern::X2XCorr)
        .def("X2PixelCorr", &PowderPattern::X2PixelCorr)
        .def("X2Pixel", &PowderPattern::X2Pixel)
        .def("STOL2X", &PowderPattern::STOL2X)
        .def("X2STOL", &PowderPattern::X2STOL)
        .def("X2XCorr", &PowderPattern::X2XCorr)
        .def("STOL2Pixel", &PowderPattern::STOL2Pixel)
        .def("UpdateDisplay", &PowderPattern::UpdateDisplay,
            &PowderPatternWrap::default_UpdateDisplay)
        .add_property("mur", &PowderPattern::GetMuR, &PowderPattern::SetMuR)
        .add_property("rw", &PowderPattern::GetRw)
        .add_property("chi2", &PowderPattern::GetChi2)
        .add_property("wavelength", &PowderPattern::GetWavelength,
                      (void (PowderPattern::*)(double)) &PowderPattern::SetWavelength)
        ;

    class_<SPGScore>("SPGScore", init<const string &, const REAL, const REAL,
            const unsigned int, bp::optional<const REAL> >
            ((bp::arg("hermann_mauguin"), bp::arg("rw"), bp::arg("gof"),
              bp::arg("nbextinct"), bp::arg("ngof")=0)))
        .def_readonly("hermann_mauguin", &SPGScore::hm)
        .def_readonly("Rw", &SPGScore::rw)
        .def_readonly("GoF", &SPGScore::gof)
        .def_readonly("nGoF", &SPGScore::ngof)
        .def("__str__", &__str__SPGScore)
        .def("__repr__", &__str__SPGScore)
        ;
    //,init<PowderPatternDiffraction *>(bp::arg("powdiff")),with_custodian_and_ward_postcall<1,2>()
    class_<SpaceGroupExplorer>("SpaceGroupExplorer", init<PowderPatternDiffraction * >
            ((bp::arg("powdiff"))) [with_custodian_and_ward<1,2>()])
        .def("Run", (SPGScore (SpaceGroupExplorer::*)(const string&, const bool, const bool, const bool, const bool))
             &SpaceGroupExplorer::Run,
             (bp::arg("spg"), bp::arg("fitprofile")=false, bp::arg("verbose")=false,
             bp::arg("restore_orig")=false, bp::arg("update_display")=false))
        .def("RunAll", &SpaceGroupExplorer::RunAll,
             (bp::arg("fitprofile_all")=false, bp::arg("verbose")=true,
             bp::arg("keep_best")=true, bp::arg("update_display")=true,
             bp::arg("fitprofile_p1")=true))
        .def("GetScores", &_GetScores)
        ;

    // This will only return a C++ PowderPattern object
    def("CreatePowderPatternFromCIF",
            (PowderPattern* (*)(bp::object input)) &_CreatePowderPatternFromCIF, bp::arg("file"),
            return_value_policy<manage_new_object>());

    // This can update a PowderPattern object with python methods
    def("CreatePowderPatternFromCIF",
            (PowderPattern* (*)(bp::object input, PowderPattern &)) &_CreatePowderPatternFromCIF, (bp::arg("file"), bp::arg("powpat")),
            return_value_policy<manage_new_object>());
}
