#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/make_iterator.h>

#include <sstream>
#include <fstream>
#include <cmath>

#undef B0
#include <ObjCryst/ObjCryst/Indexing.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
#define RAD2DEG (180./M_PI)

namespace {

// RecUnitCell::hkl2d takes a raw `REAL *derivpar` output parameter. nanobind's
// automatic None<->nullptr pointer bridging (as used elsewhere in this codebase,
// e.g. Crystal.AddScatterer's Scatterer* argument) only works for pointers to a
// type registered via nb::class_<...> -- there is no such bridging for a pointer
// to a plain arithmetic type like REAL/double, so binding
// `&RecUnitCell::hkl2d` directly rejects an explicit `derivpar=None` from Python
// with "incompatible function arguments", even though None is exactly what a
// Python caller must pass (a Python float is immutable, so there is no way for
// the derivative to be written back to it regardless of binding technology --
// this matches the pre-existing Boost.Python behavor, where derivpar was only
// ever meaningfully used from within C++ itself, e.g. LSQ fitting internals).
// Fixed by taking derivpar as nb::object and manually bridging None -> nullptr.
float _hkl2d(const RecUnitCell& r, const float h, const float k, const float l,
             nb::object derivpar, const unsigned int derivhkl)
{
    if (derivpar.is_none())
        return r.hkl2d(h, k, l, nullptr, derivhkl);
    REAL d = 0;
    return r.hkl2d(h, k, l, &d, derivhkl);
}

nb::tuple _direct_unit_cell(const RecUnitCell& r, const bool equiv, const bool degrees=false)
{
    const std::vector<float>& v = r.DirectUnitCell(equiv);
    if (degrees)
        return nb::make_tuple(v[0], v[1], v[2], v[3]*RAD2DEG, v[4]*RAD2DEG, v[5]*RAD2DEG, v[6]);
    return nb::make_tuple(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
}

nb::list _vDicVolHKL(const PeakList::hkl& h)
{
    nb::list l;
    for (const auto& hkl0 : h.vDicVolHKL)
        l.append(hkl0);
    return l;
}

std::string __str__RecUnitCell(RecUnitCell& ruc)
{
    const char* sys_names[] = {"TRICLINIC","MONOCLINIC","ORTHORHOMBIC","HEXAGONAL","RHOMBOEDRAL","TETRAGONAL","CUBIC"};
    const char cent_chars[] = "PIABCF";
    std::vector<float> d = ruc.DirectUnitCell();
    std::ostringstream oss;
    if (ruc.mNbSpurious > 0)
        oss << d[0] << " " << d[1] << " " << d[2]
            << " " << d[3]*RAD2DEG << " " << d[4]*RAD2DEG << " " << d[5]*RAD2DEG
            << " V=" << d[6] << " " << sys_names[(int)ruc.mlattice]
            << " " << cent_chars[(int)ruc.mCentering]
            << " (" << ruc.mNbSpurious << " SPURIOUS)";
    else
        oss << d[0] << " " << d[1] << " " << d[2]
            << " " << d[3]*RAD2DEG << " " << d[4]*RAD2DEG << " " << d[5]*RAD2DEG
            << " V=" << d[6] << " " << sys_names[(int)ruc.mlattice]
            << " " << cent_chars[(int)ruc.mCentering];
    return oss.str();
}

std::string __str__hkl0(PeakList::hkl0& hkl)
{
    std::ostringstream oss;
    oss << "(" << hkl.h << " " << hkl.k << " " << hkl.l << ")";
    return oss.str();
}

std::string __str__hkl(PeakList::hkl& hkl)
{
    std::ostringstream oss;
    if (hkl.isIndexed)
        oss << "Peak dobs=" << hkl.dobs << "+/-" << hkl.dobssigma
            << " iobs=" << hkl.iobs
            << " (" << hkl.h << " " << hkl.k << " " << hkl.l << ")";
    else
        oss << "Peak dobs=" << hkl.dobs << "+/-" << hkl.dobssigma
            << " iobs=" << hkl.iobs << " (? ? ?)";
    return oss.str();
}

void _DicVolGag(CellExplorer& ex, const float minScore, const unsigned int minDepth,
                const float stopOnScore, const unsigned int stopOnDepth, const bool verbose=true)
{
    CaptureStdOut gag;
    if (verbose) gag.release();
    ex.DicVol(minScore, minDepth, stopOnScore, stopOnDepth);
}

nb::list _GetSolutions(CellExplorer& ex)
{
    nb::list l;
    for (const auto& p : ex.GetSolutions())
        l.append(nb::make_tuple(nb::cast(p.first), p.second));
    return l;
}

struct PeakListNB : PeakList {
    PeakListNB() : PeakList() {}
    PeakListNB(const PeakList& p) : PeakList(p) {}

    void nb_ImportDhklDSigmaIntensity(nb::object input, const float defaultsigma)
    {
        CaptureStdOut gag;
        if (nb::isinstance<nb::str>(input)) {
            std::string path = nb::cast<std::string>(input);
            std::ifstream is(path);
            this->PeakList::ImportDhklDSigmaIntensity(is, defaultsigma);
        } else {
            std::string s = read_pyfile_to_string(input);
            std::istringstream is(s);
            this->PeakList::ImportDhklDSigmaIntensity(is, defaultsigma);
        }
    }

    void nb_ImportDhklIntensity(nb::object input)
    {
        CaptureStdOut gag;
        if (nb::isinstance<nb::str>(input)) {
            std::string path = nb::cast<std::string>(input);
            std::ifstream is(path);
            this->PeakList::ImportDhklIntensity(is);
        } else {
            std::string s = read_pyfile_to_string(input);
            std::istringstream is(s);
            this->PeakList::ImportDhklIntensity(is);
        }
    }

    void nb_ImportDhkl(nb::object input)
    {
        CaptureStdOut gag;
        if (nb::isinstance<nb::str>(input)) {
            std::string path = nb::cast<std::string>(input);
            std::ifstream is(path);
            this->PeakList::ImportDhkl(is);
        } else {
            std::string s = read_pyfile_to_string(input);
            std::istringstream is(s);
            this->PeakList::ImportDhkl(is);
        }
    }

    void nb_Import2ThetaIntensity(nb::object input, const float wavelength)
    {
        CaptureStdOut gag;
        if (nb::isinstance<nb::str>(input)) {
            std::string path = nb::cast<std::string>(input);
            std::ifstream is(path);
            this->PeakList::Import2ThetaIntensity(is, wavelength);
        } else {
            std::string s = read_pyfile_to_string(input);
            std::istringstream is(s);
            this->PeakList::Import2ThetaIntensity(is, wavelength);
        }
    }

    void nb_ExportDhklDSigmaIntensity(nb::object output)
    {
        if (nb::isinstance<nb::str>(output)) {
            std::string path = nb::cast<std::string>(output);
            std::ofstream out(path);
            this->PeakList::ExportDhklDSigmaIntensity(out);
        } else {
            std::ostringstream oss;
            this->PeakList::ExportDhklDSigmaIntensity(oss);
            output.attr("write")(oss.str());
        }
    }

    void set_dobs_list(nb::list& l)
    {
        this->GetPeakList().clear();
        for (auto h : l)
            this->AddPeak(nb::cast<float>(h));
    }

    unsigned int Length() const { return (unsigned int)this->GetPeakList().size(); }

    nb::list getPeakList()
    {
        nb::list l;
        for (const auto& hkl : this->GetPeakList())
            l.append(hkl);
        return l;
    }
};

} // namespace

// External-linkage wrapper declared in helpers_nb.hpp -- see the comment
// there. Must live outside the anonymous namespace above (which is where
// PeakListNB itself is defined) so other translation units can call it.
nb::object wrap_peaklist(const PeakList& pl)
{
    return nb::cast(PeakListNB(pl));
}

void wrap_indexing(nb::module_& m)
{
    nb::enum_<CrystalSystem>(m, "CrystalSystem")
        .value("TRICLINIC",   TRICLINIC)
        .value("MONOCLINIC",  MONOCLINIC)
        // Ported forward from upstream main (spelling fix, pre-commit
        // auto-fix commit): the C++ enum is (and always was) ORTHORHOMBIC --
        // ORTHOROMBIC was a Python-binding-only misspelling. Keep both
        // names bound to the same value for backward compatibility.
        .value("ORTHORHOMBIC", ORTHORHOMBIC)
        .value("ORTHOROMBIC", ORTHORHOMBIC)
        .value("HEXAGONAL",   HEXAGONAL)
        .value("RHOMBOEDRAL", RHOMBOEDRAL)
        .value("TETRAGONAL",  TETRAGONAL)
        .value("CUBIC",       CUBIC)
        .export_values()
        ;

    nb::enum_<CrystalCentering>(m, "CrystalCentering")
        .value("LATTICE_P", LATTICE_P)
        .value("LATTICE_I", LATTICE_I)
        .value("LATTICE_A", LATTICE_A)
        .value("LATTICE_B", LATTICE_B)
        .value("LATTICE_C", LATTICE_C)
        .value("LATTICE_F", LATTICE_F)
        .export_values()
        ;

    m.def("EstimateCellVolume", &EstimateCellVolume,
          nb::arg("dmin"), nb::arg("dmax"), nb::arg("nbrefl"),
          nb::arg("system"), nb::arg("centering"), nb::arg("kappa") = 1.f);

    nb::class_<RecUnitCell>(m, "RecUnitCell")
        .def(nb::init<const float, const float, const float, const float,
                      const float, const float, const float, CrystalSystem,
                      CrystalCentering, const unsigned int>(),
             nb::arg("zero") = 0.f, nb::arg("par0") = 0.f, nb::arg("par1") = 0.f,
             nb::arg("par2") = 0.f, nb::arg("par3") = 0.f, nb::arg("par4") = 0.f,
             nb::arg("par5") = 0.f, nb::arg("lattice") = CUBIC,
             nb::arg("cent") = LATTICE_P, nb::arg("nbspurious") = 0u)
        .def(nb::init<const RecUnitCell&>(), nb::arg("old"))
        .def("hkl2d", &_hkl2d,
             nb::arg("h"), nb::arg("k"), nb::arg("l"),
             nb::arg("derivpar") = nb::none(), nb::arg("derivhkl") = 0)
        .def("DirectUnitCell", &_direct_unit_cell,
             nb::arg("equiv") = false, nb::arg("degrees") = false)
        .def_ro("mlattice",    &RecUnitCell::mlattice)
        .def_ro("lattice",     &RecUnitCell::mlattice)
        .def_ro("mCentering",  &RecUnitCell::mCentering)
        .def_ro("centering",   &RecUnitCell::mCentering)
        .def_ro("nb_spurious", &RecUnitCell::mNbSpurious)
        .def_ro("mNbSpurious", &RecUnitCell::mNbSpurious)
        .def("__str__",  &__str__RecUnitCell)
        .def("__repr__", &__str__RecUnitCell)
        ;

    nb::class_<PeakList::hkl0>(m, "PeakList_hkl0")
        .def(nb::init<const int, const int, const int>(),
             nb::arg("h") = 0, nb::arg("k") = 0, nb::arg("l") = 0)
        .def_ro("h", &PeakList::hkl0::h)
        .def_ro("k", &PeakList::hkl0::k)
        .def_ro("l", &PeakList::hkl0::l)
        .def("__str__",  &__str__hkl0)
        .def("__repr__", &__str__hkl0)
        ;

    nb::class_<PeakList::hkl>(m, "PeakList_hkl")
        .def(nb::init<const float, const float, const float, const float,
                      const int, const int, const int, const float>(),
             nb::arg("dobs") = 1.0f, nb::arg("iobs") = 0.f,
             nb::arg("dobssigma") = 0.f, nb::arg("iobssigma") = 0.f,
             nb::arg("h") = 0, nb::arg("k") = 0, nb::arg("l") = 0,
             nb::arg("d2calc") = 0.f)
        .def_ro("dobs",      &PeakList::hkl::dobs)
        .def_ro("dobssigma", &PeakList::hkl::dobssigma)
        .def_ro("d2obs",     &PeakList::hkl::d2obs)
        .def_ro("d2obsmin",  &PeakList::hkl::d2obsmin)
        .def_ro("d2obsmax",  &PeakList::hkl::d2obsmax)
        .def_ro("iobs",      &PeakList::hkl::iobs)
        .def_ro("iobssigma", &PeakList::hkl::iobssigma)
        .def_ro("h",         &PeakList::hkl::h)
        .def_ro("k",         &PeakList::hkl::k)
        .def_ro("l",         &PeakList::hkl::l)
        .def_ro("isIndexed", &PeakList::hkl::isIndexed)
        .def_prop_ro("vDicVolHKL", &_vDicVolHKL)
        .def_ro("isSpurious",&PeakList::hkl::isSpurious)
        .def_ro("stats",     &PeakList::hkl::stats)
        .def_ro("d2calc",    &PeakList::hkl::d2calc)
        .def_ro("d2diff",    &PeakList::hkl::d2diff)
        .def("__str__",  &__str__hkl)
        .def("__repr__", &__str__hkl)
        ;

    // PeakListNB : public PeakList is plain (non-virtual) inheritance --
    // perfectly safe to declare as a nanobind base (unlike the virtual-
    // inheritance cases in §1 of nanobind_migration_notes.md) -- but it was
    // previously registered with NO declared base at all
    // (`nb::class_<PeakListNB>(m, "PeakList")`), which meant nanobind had no
    // known relationship between the two types. That broke
    // `CellExplorer(pl, ...)` (nb::init<const PeakList&, ...>()): passing a
    // Python "PeakList" (i.e. a PeakListNB instance, e.g. from
    // PowderPattern.FindPeaks()) failed to convert to `const PeakList&`
    // with a generic "incompatible function arguments" error. `PeakList`
    // itself has no other nanobind bindings of its own (nothing but
    // PeakListNB is ever exposed to Python), so it is registered here under
    // an internal name purely so the base relationship can be declared.
    nb::class_<PeakList>(m, "_PeakListBase");

    nb::class_<PeakListNB, PeakList>(m, "PeakList")
        .def(nb::init<>())
        .def(nb::init<const PeakList&>())
        .def("ImportDhklDSigmaIntensity", &PeakListNB::nb_ImportDhklDSigmaIntensity,
             nb::arg("file"), nb::arg("defaultsigma") = 0.001f)
        .def("ImportDhklIntensity",       &PeakListNB::nb_ImportDhklIntensity,  nb::arg("file"))
        .def("ImportDhkl",                &PeakListNB::nb_ImportDhkl,           nb::arg("file"))
        .def("Import2ThetaIntensity",     &PeakListNB::nb_Import2ThetaIntensity,
             nb::arg("file"), nb::arg("wavelength"))
        .def("ExportDhklDSigmaIntensity", &PeakListNB::nb_ExportDhklDSigmaIntensity, nb::arg("file"))
        .def("Simulate", &PeakList::Simulate,
             nb::arg("zero"), nb::arg("a"), nb::arg("b"), nb::arg("c"),
             nb::arg("alpha"), nb::arg("beta"), nb::arg("gamma"), nb::arg("deg"),
             nb::arg("nb") = 20, nb::arg("nbspurious") = 0, nb::arg("sigma") = 0.f,
             nb::arg("percentMissing") = 0.f, nb::arg("verbose") = false,
             // "merge" is new in the vendored ObjCryst++ core since this
             // binding was originally written (not present when the old
             // migration branch's submodule was pinned) -- merges observed
             // lines with near-identical d-spacing before picking the
             // largest nb lines. Default false preserves old behavior.
             nb::arg("merge") = false)
        .def("GetPeakList", &PeakListNB::getPeakList)
        .def("resize", [](PeakListNB& pl, unsigned int nb){ pl.GetPeakList().resize(nb); },
             nb::arg("nb") = 20)
        .def("clear", [](PeakListNB& pl){ pl.GetPeakList().clear(); })
        .def("set_dobs_list", &PeakListNB::set_dobs_list, nb::arg("dobs"))
        .def("__len__", &PeakListNB::Length)
        .def("__iter__", [](PeakListNB& pl){
            auto& v = pl.GetPeakList();
            return nb::make_iterator(nb::type<PeakListNB>(), "peak_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        ;

    nb::class_<CellExplorer, RefinableObj>(m, "CellExplorer")
        .def(nb::init<const PeakList&, const CrystalSystem, const unsigned int>(),
             nb::arg("dhkl"), nb::arg("lattice"), nb::arg("nbSpurious") = 0u,
             nb::keep_alive<1,2>())
        .def("SetLengthMinMax", &CellExplorer::SetLengthMinMax)
        .def("SetAngleMinMax",  &CellExplorer::SetAngleMinMax)
        .def("SetVolumeMinMax", &CellExplorer::SetVolumeMinMax)
        .def("SetNbSpurious",   &CellExplorer::SetNbSpurious)
        .def("SetD2Error",      &CellExplorer::SetD2Error)
        .def("SetMinMaxZeroShift", &CellExplorer::SetMinMaxZeroShift)
        .def("SetCrystalSystem",   &CellExplorer::SetCrystalSystem)
        .def("SetCrystalCentering",&CellExplorer::SetCrystalCentering)
        .def("Print",    &CellExplorer::Print)
        .def("DicVol",   &_DicVolGag,
             nb::arg("minScore") = 10.f, nb::arg("minDepth") = 10u,
             nb::arg("stopOnScore") = 50.f, nb::arg("stopOnDepth") = 6u,
             nb::arg("verbose") = true)
        .def("ReduceSolutions", &CellExplorer::ReduceSolutions,
             nb::arg("updateReportThreshold") = false)
        .def("GetBestScore", &CellExplorer::GetBestScore)
        .def("GetSolutions", &_GetSolutions)
        .def("__iter__", [](CellExplorer& ex){
            auto& v = ex.GetSolutions();
            return nb::make_iterator(nb::type<CellExplorer>(), "sol_iter", v.begin(), v.end());
        }, nb::keep_alive<0,1>())
        ;
}
