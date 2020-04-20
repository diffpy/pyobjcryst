/*****************************************************************************
*
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::Indexing.
*
* Changes from Indexing:
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#undef B0

#include <string>

#include <ObjCryst/ObjCryst/Indexing.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "python_streambuf.hpp"
#include "helpers.hpp"


namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define RAD2DEG (180./M_PI)



namespace {

bp::tuple _direct_unit_cell(const RecUnitCell &r, const bool equiv, const bool degrees=false)
{
  const std::vector<float> &v = r.DirectUnitCell(equiv);
  if(degrees)
    return bp::make_tuple(v[0], v[1], v[2], v[3]*RAD2DEG, v[4]*RAD2DEG, v[5]*RAD2DEG, v[6]);
  return bp::make_tuple(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
}

bp::list _vDicVolHKL(const PeakList::hkl& h)
{
    return listToPyList< PeakList::hkl0 > (h.vDicVolHKL);
}

std::string __str__RecUnitCell(RecUnitCell& ruc)
{
    std::string sys;
    switch(ruc.mlattice)
    {
      case TRICLINIC:sys="TRICLINIC"; break;
      case MONOCLINIC:sys="MONOCLINIC"; break;
      case ORTHOROMBIC:sys="ORTHOROMBIC"; break;
      case HEXAGONAL:sys="HEXAGONAL"; break;
      case RHOMBOEDRAL:sys="RHOMBOEDRAL"; break;
      case TETRAGONAL:sys="TETRAGONAL"; break;
      case CUBIC:sys="CUBIC"; break;
    }

    char centc;
    switch(ruc.mCentering)
    {
      case LATTICE_P:centc='P'; break;
      case LATTICE_I:centc='I'; break;
      case LATTICE_A:centc='A'; break;
      case LATTICE_B:centc='B'; break;
      case LATTICE_C:centc='C'; break;
      case LATTICE_F:centc='F'; break;
    }

    std::vector<float> d = ruc.DirectUnitCell();

    if(ruc.mNbSpurious>0)
      return (boost::format("%5.2f %5.2f %5.2f %5.1f %5.1f %5.1f V=%4.0f %s %c (%d SPURIOUS)")
                           % d[0] % d[1] % d[2] % (d[3]*RAD2DEG) % (d[4]*RAD2DEG)
                           % (d[5]*RAD2DEG) % d[6] % sys % centc % ruc.mNbSpurious).str();

    return (boost::format("%5.2f %5.2f %5.2f %5.1f %5.1f %5.1f V=%4.0f %s %c")
                         % d[0] % d[1] % d[2] % (d[3]*RAD2DEG) % (d[4]*RAD2DEG)
                         % (d[5]*RAD2DEG) % d[6] % sys % centc).str();
}

std::string __str__hkl0(PeakList::hkl0& hkl)
{
    return (boost::format("(%2d %2d %2d)") % hkl.h % hkl.k % hkl.l).str();
}

std::string __str__hkl(PeakList::hkl& hkl)
{
    if(hkl.isIndexed)
      return (boost::format("Peak dobs=%7.5f+/-%7.5f iobs=%6e (%2d %2d %2d))")
                           % hkl.dobs % hkl.dobssigma % hkl.iobs
                           % hkl.h % hkl.k % hkl.l).str();

    return (boost::format("Peak dobs=%7.5f+/-%7.5f iobs=%6e (? ? ?))")
                     % hkl.dobs % hkl.dobssigma % hkl.iobs).str();
}

class PeakListWrap : public PeakList, public wrapper<PeakList>
{
    public:
      PeakListWrap():PeakList() {}

      PeakListWrap(const PeakList &p):PeakList(p) {}

      void ImportDhklDSigmaIntensity(bp::object input, const float defaultsigma)
      {
          CaptureStdOut gag;
          const std::string cname = boost::python::extract<std::string>
                                     (input.attr("__class__").attr("__name__"));
          if(cname.compare("str")==0)
          { // Filename
            boost::filesystem::path p{extract<std::string>(input)};
            boost::filesystem::ifstream is(p);
            this->PeakList::ImportDhklDSigmaIntensity(is, defaultsigma);
          }
          else
          { // Python file object
            boost_adaptbx::python::streambuf sbuf(input);
            boost_adaptbx::python::streambuf::istream is(sbuf);
            this->PeakList::ImportDhklDSigmaIntensity(is, defaultsigma);
          }
      }


      void ImportDhklIntensity(bp::object input)
      {
          CaptureStdOut gag;
          const std::string cname = boost::python::extract<std::string>
                                     (input.attr("__class__").attr("__name__"));
          if(cname.compare("str")==0)
          { // Filename
            boost::filesystem::path p{extract<std::string>(input)};
            boost::filesystem::ifstream is(p);
            this->PeakList::ImportDhklIntensity(is);
          }
          else
          {// Python file object
            boost_adaptbx::python::streambuf sbuf(input);
            boost_adaptbx::python::streambuf::istream is(sbuf);
            this->PeakList::ImportDhklIntensity(is);
          }
      }

      void default_ImportDhkl(bp::object input)
      {
          CaptureStdOut gag;
          const std::string cname = boost::python::extract<std::string>
                                     (input.attr("__class__").attr("__name__"));
          if(cname.compare("str")==0)
          { // Filename
            boost::filesystem::path p{extract<std::string>(input)};
            boost::filesystem::ifstream is(p);
            this->PeakList::ImportDhkl(is);
          }
          else
          {// Python file object
            boost_adaptbx::python::streambuf sbuf(input);
            boost_adaptbx::python::streambuf::istream is(sbuf);
            this->PeakList::ImportDhkl(is);
          }
      }

      void Import2ThetaIntensity(bp::object input, const float wavelength)
      {
          CaptureStdOut gag;
          const std::string cname = boost::python::extract<std::string>
                                     (input.attr("__class__").attr("__name__"));
          if(cname.compare("str")==0)
          { // Filename
            boost::filesystem::path p{extract<std::string>(input)};
            boost::filesystem::ifstream is(p);
            this->PeakList::Import2ThetaIntensity(is, wavelength);
          }
          else
          {// Python file object
            boost_adaptbx::python::streambuf sbuf(input);
            boost_adaptbx::python::streambuf::istream is(sbuf);
            this->PeakList::Import2ThetaIntensity(is, wavelength);
          }
      }

      void ExportDhklDSigmaIntensity(bp::object output)
      {
          CaptureStdOut gag;
          const std::string cname = bp::extract<std::string>
                                     (output.attr("__class__").attr("__name__"));
          if(cname.compare("str")==0)
          { // Filename
            boost::filesystem::path p{extract<std::string>(output)};
            boost::filesystem::ofstream out(p);
            this->PeakList::ExportDhklDSigmaIntensity(out);
          }
          else
          {// Python file object
            boost_adaptbx::python::streambuf sbuf(output);
            boost_adaptbx::python::streambuf::ostream out(sbuf);
            this->PeakList::ExportDhklDSigmaIntensity(out);
          }
      }

      void set_dobs_list(bp::list &l)
      {
          this->GetPeakList().clear();
          bp::ssize_t n = bp::len(l);
          for(bp::ssize_t i=0;i<n;i++)
            this->AddPeak(bp::extract<float>(l[i]));
      }

      unsigned int Length() const
      {return this->GetPeakList().size();}

      std::vector<PeakList::hkl>::const_iterator GetPeakIterBegin()
      {return this->GetPeakList().begin();}

      std::vector<PeakList::hkl>::const_iterator GetPeakIterEnd()
      {return this->GetPeakList().end();}

      void resize(const unsigned int nb)
      {this->GetPeakList().resize(nb);}

      void clear()
      {this->GetPeakList().clear();}

      bp::list _getPeakList()
      {
          return vectorToPyList< PeakList::hkl > (this->GetPeakList());
      }

};

std::list<std::pair<RecUnitCell,float> >::const_iterator _GetSolutionsIterBegin(const CellExplorer& c)
{return c.GetSolutions().begin();}

std::list<std::pair<RecUnitCell,float> >::const_iterator _GetSolutionsIterEnd(const CellExplorer& c)
{return c.GetSolutions().end();}

void _DicVolGag(CellExplorer &ex, const float minScore,const unsigned int minDepth,
                const float stopOnScore,const unsigned int stopOnDepth, const bool verbose=true)
{
    CaptureStdOut gag;
    if(verbose) gag.release();
    ex.DicVol(minScore, minDepth, stopOnScore, stopOnDepth);
}

// Custom converter
struct pair_ruc_float_to_tuple
{
    static PyObject* convert(std::pair<RecUnitCell, float> const &p)
    {
        bp::object tpl = bp::make_tuple(bp::ptr(&p.first), p.second);
        PyObject* rv = tpl.ptr();
        return bp::incref(rv);
    }

    static PyTypeObject const* get_pytype()
    {
        return &PyTuple_Type;
    }
};

bp::list _GetSolutions(CellExplorer &ex)
{  // See containerToPyList
  bp::list l;

  for(std::list<std::pair<RecUnitCell,float> >::const_iterator pos = ex.GetSolutions().begin();
      pos != ex.GetSolutions().end(); ++pos)
    l.append(*pos);

  return l;
}

} // namespace


void wrap_indexing()
{
    enum_<CrystalSystem>("CrystalSystem")
        .value("TRICLINIC", TRICLINIC)
        .value("MONOCLINIC", MONOCLINIC)
        .value("ORTHOROMBIC", ORTHOROMBIC)
        .value("HEXAGONAL", HEXAGONAL)
        .value("RHOMBOEDRAL", RHOMBOEDRAL)
        .value("TETRAGONAL", TETRAGONAL)
        .value("CUBIC", CUBIC)
        ;

    enum_<CrystalCentering>("CrystalCentering")
        .value("LATTICE_P", LATTICE_P)
        .value("LATTICE_I", LATTICE_I)
        .value("LATTICE_A", LATTICE_A)
        .value("LATTICE_B", LATTICE_B)
        .value("LATTICE_C", LATTICE_C)
        .value("LATTICE_F", LATTICE_F)
        ;

    def("EstimateCellVolume", &EstimateCellVolume,
        (bp::arg("dmin"), bp::arg("dmax"), bp::arg("nbrefl"), bp::arg("system"),
        bp::arg("centering"), bp::arg("kappa")=1.));

    class_<RecUnitCell> ("RecUnitCell")
        .def(init<const float, const float, const float, const float,
                  const float, const float, const float, CrystalSystem,
                  CrystalCentering, const unsigned int>(
            (bp::arg("zero")=0, bp::arg("par0")=0, bp::arg("par1")=0,
            bp::arg("par2")=0, bp::arg("par3")=0, bp::arg("par4")=0,
            bp::arg("par5")=0, bp::arg("lattice")=CUBIC,
            bp::arg("cent")=LATTICE_P, bp::arg("nbspurious")=0)))
        .def(init<const RecUnitCell&>(bp::arg("old")))
        .def("hkl2d", &RecUnitCell::hkl2d,
             (bp::arg("h"), bp::arg("k"), bp::arg("l"),
              bp::arg("derivpar")=NULL, bp::arg("derivhkl")=0))
        .def("DirectUnitCell", &_direct_unit_cell,
             (bp::arg("equiv")=false, bp::arg("degrees")=false))
        //.def_readonly("par", &RecUnitCell::par)
        .def_readonly("mlattice", &RecUnitCell::mlattice)
        .def_readonly("mCentering", &RecUnitCell::mCentering)
        .def_readonly("mNbSpurious", &RecUnitCell::mNbSpurious)
        .def("__str__", &__str__RecUnitCell)
        .def("__repr__", &__str__RecUnitCell)
        ;

    // Avoid the name hkl which may conflict somewhere else
    class_<PeakList::hkl0> ("PeakList_hkl0")
        .def(init<const int, const int, const int>(
            (bp::arg("h")=0, bp::arg("k")=0, bp::arg("l")=0)))
        .def_readonly("h", &PeakList::hkl0::h)
        .def_readonly("k", &PeakList::hkl0::k)
        .def_readonly("l", &PeakList::hkl0::l)
        .def("__str__", &__str__hkl0)
        .def("__repr__", &__str__hkl0)
    ;

    // Avoid the name hkl which may conflict somewhere else
    class_<PeakList::hkl> ("PeakList_hkl")
        .def(init<const float, const float, const float, const float,
                  const int, const int, const int, const float>(
            (bp::arg("dobs")=1.0, bp::arg("iobs")=0, bp::arg("dobssigma")=0,
            bp::arg("iobssigma")=0, bp::arg("h")=0, bp::arg("k")=0,
            bp::arg("l")=0, bp::arg("d2calc")=0)))
        .def_readonly("dobs", &PeakList::hkl::dobs)
        .def_readonly("dobssigma", &PeakList::hkl::dobssigma)
        .def_readonly("d2obs", &PeakList::hkl::d2obs)
        .def_readonly("d2obsmin", &PeakList::hkl::d2obsmin)
        .def_readonly("d2obsmax", &PeakList::hkl::d2obsmax)
        .def_readonly("iobs", &PeakList::hkl::iobs)
        .def_readonly("iobssigma", &PeakList::hkl::iobssigma)
        .def_readonly("h", &PeakList::hkl::h)
        .def_readonly("k", &PeakList::hkl::k)
        .def_readonly("l", &PeakList::hkl::l)
        .def_readonly("isIndexed", &PeakList::hkl::isIndexed)
        .add_property("vDicVolHKL", &_vDicVolHKL)
        .def_readonly("isSpurious", &PeakList::hkl::isSpurious)
        .def_readonly("stats", &PeakList::hkl::stats)
        .def_readonly("d2calc", &PeakList::hkl::d2calc)
        .def_readonly("d2diff", &PeakList::hkl::d2diff)
        .def("__str__", &__str__hkl)
        .def("__repr__", &__str__hkl)
    ;

    class_<PeakListWrap> ("PeakList")
        .def("ImportDhklDSigmaIntensity",
                &PeakListWrap::ImportDhklDSigmaIntensity,
                (bp::arg("file"), bp::arg("defaultsigma")=0.001))
        .def("ImportDhklIntensity",
                &PeakListWrap::ImportDhklIntensity,
                bp::arg("file"))
        .def("ImportDhkl",
                &PeakListWrap::ImportDhkl,
                bp::arg("file"))
        .def("Import2ThetaIntensity",
                &PeakListWrap::Import2ThetaIntensity,
                (bp::arg("file"), bp::arg("wavelength")))
        .def("ExportDhklDSigmaIntensity",
                &PeakListWrap::ExportDhklDSigmaIntensity,
                bp::arg("file"))
        .def("Simulate", &PeakList::Simulate,
            (bp::arg("zero"), bp::arg("a"), bp::arg("b"), bp::arg("c"),
            bp::arg("alpha"), bp::arg("beta"), bp::arg("gamma"), bp::arg("deg"),
            bp::arg("nb")=20, bp::arg("nbspurious")=0, bp::arg("sigma")=0,
            bp::arg("percentMissing")=0, bp::arg("verbose")=false))
        .def("GetPeakList", &PeakListWrap::_getPeakList,
            with_custodian_and_ward_postcall<1,0>())
        // Python only
        .def("resize", &PeakListWrap::resize, bp::arg("nb")=20)
        .def("clear", &PeakListWrap::clear)
        .def("set_dobs_list", &PeakListWrap::set_dobs_list, (bp::arg("dobs")))
        .def("__len__", &PeakListWrap::Length)
        .def("__iter__", range<return_value_policy<reference_existing_object> >
                   (&PeakListWrap::GetPeakIterBegin, &PeakListWrap::GetPeakIterEnd))
        ;

    // Register converter
    boost::python::to_python_converter<const std::pair<RecUnitCell, float>,
                                       pair_ruc_float_to_tuple>();

    class_<CellExplorer , bases<RefinableObj>, boost::noncopyable>("CellExplorer",
        init<const PeakList&, const CrystalSystem, const unsigned int>(
             (bp::arg("dhkl"), bp::arg("lattice"), bp::arg("nbSpurious")=0))
             [with_custodian_and_ward<1,2>()])
        .def("SetLengthMinMax", &CellExplorer::SetLengthMinMax)
        .def("SetAngleMinMax", &CellExplorer::SetAngleMinMax)
        .def("SetVolumeMinMax", &CellExplorer::SetVolumeMinMax)
        .def("SetNbSpurious", &CellExplorer::SetNbSpurious)
        .def("SetD2Error", &CellExplorer::SetD2Error)
        .def("SetMinMaxZeroShift", &CellExplorer::SetMinMaxZeroShift)
        .def("SetCrystalSystem", &CellExplorer::SetCrystalSystem)
        .def("SetCrystalCentering", &CellExplorer::SetCrystalCentering)
        .def("Print", &CellExplorer::Print)
        .def("DicVol", &_DicVolGag,
             (bp::arg("minScore")=10, bp::arg("minScore")=10, bp::arg("stopOnScore")=50,
              bp::arg("stopOnDepth")=6, bp::arg("verbose")=true))
        .def("ReduceSolutions", &CellExplorer::ReduceSolutions,
             bp::arg("updateReportThreshold")=false)
        .def("GetBestScore", &CellExplorer::GetBestScore)
        .def("GetSolutions", &_GetSolutions)
        // Python only
        .def("__iter__", range<return_value_policy<reference_existing_object> >
             (&_GetSolutionsIterBegin, &_GetSolutionsIterEnd))
        ;
}
