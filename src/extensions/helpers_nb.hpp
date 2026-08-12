/*
 * pyobjcryst nanobind port
 * helpers_nb.hpp — utilities analogous to helpers.hpp but without boost::python
 */

#ifndef HELPERS_NB_HPP
#define HELPERS_NB_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <string>
#include <sstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <set>

#include <ObjCryst/CrystVector/CrystVector.h>
#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace nb = nanobind;

typedef std::numeric_limits<double> doublelim;

// ---------------------------------------------------------------------------
// wrap_peaklist — wrap a plain ObjCryst::PeakList (returned by value from
// several C++ APIs, e.g. PowderPattern::FindPeaks()) into the Python
// "PeakList" type. The actual nanobind-registered type is PeakListNB, a
// thin subclass defined in an anonymous namespace in nb_indexing.cpp (it
// needs extra methods not present on the plain C++ PeakList) -- nanobind
// has no registered Python type for the base PeakList itself, so returning
// one directly fails with "Unable to convert function return value to a
// Python type!". This function is implemented (with external linkage) in
// nb_indexing.cpp so other translation units (e.g. nb_powderpattern.cpp)
// can convert a PeakList without needing PeakListNB's definition.
// ---------------------------------------------------------------------------
namespace ObjCryst { class PeakList; }
nb::object wrap_peaklist(const ObjCryst::PeakList& pl);

// ---------------------------------------------------------------------------
// extractRefinableObjArg — resolve an arbitrary Python object, known only to
// be "some RefinableObj-derived instance", to a C++ RefinableObj&.
//
// Several concrete classes (ScatteringData, PowderPatternComponent,
// Scatterer, ScatteringPower, and the ScatteringPowerAtom leaf) inherit
// RefinableObj VIRTUALLY and are therefore (correctly, see §1 in
// nanobind_migration_notes.md) registered in nanobind WITHOUT declaring
// RefinableObj as their base. That means functions which generically accept
// "any RefinableObj" -- MonteCarloObj::AddRefinableObj,
// LSQNumObj::SetRefinedObj, RefinableObj::AddPar(RefinableObj&, bool), and
// any future one -- have no automatic nanobind conversion path for a Python
// DiffractionDataSingleCrystal, PowderPatternBackground, Atom, Molecule,
// ScatteringPowerSphere, or ScatteringPowerAtom argument, even though a
// plain C++ static_cast<RefinableObj&> on the same object is always
// perfectly well-defined (virtual inheritance is exactly what lets the
// compiler resolve it correctly at compile time -- the problem is only ever
// nanobind's own offset-table machinery, never a real static_cast/
// dynamic_cast). Implemented (with external linkage) in nb_refinableobj.cpp,
// since that is the one file that reasonably includes every intermediate
// type's header without looking out of place.
// ---------------------------------------------------------------------------
namespace ObjCryst { class RefinableObj; }
ObjCryst::RefinableObj& extractRefinableObjArg(nb::object obj);

// ---------------------------------------------------------------------------
// MuteObjCrystUserInfo — suppress ObjCryst++ fpObjCrystInformUser during
// construction/parsing operations.
// ---------------------------------------------------------------------------
class MuteObjCrystUserInfo
{
public:
    MuteObjCrystUserInfo() :
        msave_info_func(ObjCryst::fpObjCrystInformUser)
    {
        ObjCryst::fpObjCrystInformUser = [](const std::string&) {};
    }

    ~MuteObjCrystUserInfo() { release(); }

    void release()
    {
        if (msave_info_func) {
            ObjCryst::fpObjCrystInformUser = msave_info_func;
            msave_info_func = nullptr;
        }
    }

private:
    void (*msave_info_func)(const std::string&);
};

// ---------------------------------------------------------------------------
// CaptureStdOut — redirect std::cout to a string buffer
// ---------------------------------------------------------------------------
class CaptureStdOut
{
public:
    CaptureStdOut() : msave(std::cout.rdbuf()) {
        std::cout.rdbuf(mss.rdbuf());
    }
    ~CaptureStdOut() { release(); }

    std::string str() const { return mss.str(); }

    void release() {
        if (msave) { std::cout.rdbuf(msave); msave = nullptr; }
    }

private:
    std::ostringstream mss;
    std::streambuf* msave;
};

// ---------------------------------------------------------------------------
// __str__ helper — call obj.Print(), return captured output
// ---------------------------------------------------------------------------
template <class T>
std::string obj_str(const T& obj)
{
    CaptureStdOut outbuf;
    obj.Print();
    outbuf.release();
    std::string s = outbuf.str();
    // strip trailing newline
    while (!s.empty() && s.back() == '\n') s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------
// read_pyfile_to_string — read the full contents of a Python file-like
// object into a std::string, regardless of whether it was opened in text
// mode (file.read() -> str) or binary mode (file.read() -> bytes).
//
// Callers throughout the codebase do `open(file, "rb")` (matching the old
// Boost.Python python_streambuf convention), so `.read()` returns `bytes`.
// A plain `nb::cast<std::string>(input.attr("read")())` only accepts `str`
// and throws (cast_error, reported to Python as "std::bad_cast") on bytes.
// ---------------------------------------------------------------------------
inline std::string read_pyfile_to_string(nb::object input)
{
    nb::object data = input.attr("read")();
    if (nb::isinstance<nb::bytes>(data)) {
        nb::bytes b = nb::cast<nb::bytes>(data);
        return std::string(b.c_str(), b.size());
    }
    return nb::cast<std::string>(data);
}

// ---------------------------------------------------------------------------
// Index checking (supports negative indexing)
// ---------------------------------------------------------------------------
enum NegativeIndexFlag { POSITIVE, ALLOW_NEGATIVE };

inline int check_index(int idx, int size, NegativeIndexFlag nflag = POSITIVE)
{
    if (nflag == ALLOW_NEGATIVE && idx < 0) idx += size;
    if (idx < 0 || idx >= size)
        throw nb::index_error("index out of range");
    return idx;
}

// ---------------------------------------------------------------------------
// CrystVector_REAL / CrystMatrix_REAL <-> numpy array (copies data to double)
//
// NOTE: these must use CrystVector_REAL / CrystMatrix_REAL (i.e. CrystVector<REAL>),
// NOT a hardcoded CrystVector<float>. REAL is defined project-wide as `double`
// (see CMakeLists.txt: target_compile_definitions(ObjCryst PUBLIC REAL=double),
// matching the historical SConstruct build). Hardcoding <float> here silently
// round-trips every vector/matrix through a lossy CrystVector<T>::operator
// CrystVector<U>() conversion (and fails to compile at all for the non-const
// write path), which is likely the source of several nanobind-port test
// failures (bad_cast / precision mismatches) — REAL must stay in sync here.
// ---------------------------------------------------------------------------
using nb_array_1d = nb::ndarray<nb::numpy, double, nb::shape<-1>>;
using nb_array_2d = nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>;

inline nb_array_1d crystvec_to_array(const CrystVector_REAL& cv)
{
    size_t n = cv.numElements();
    double* data = new double[n];
    for (size_t i = 0; i < n; ++i) data[i] = cv(i);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb_array_1d(data, {n}, owner);
}

inline nb_array_2d crystmat_to_array(const CrystMatrix_REAL& cm)
{
    size_t rows = cm.rows(), cols = cm.cols();
    double* data = new double[rows * cols];
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i * cols + j] = cm(i, j);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb_array_2d(data, {rows, cols}, owner);
}

// ---------------------------------------------------------------------------
// numpy array → CrystVector_REAL
// ---------------------------------------------------------------------------
inline void assign_crystvec(CrystVector_REAL& cv, nb_array_1d arr)
{
    size_t n = arr.shape(0);
    cv.resize(n);
    for (size_t i = 0; i < n; ++i) cv(i) = static_cast<REAL>(arr(i));
}

// ---------------------------------------------------------------------------
// assignCrystVector — fill CrystVector<REAL> from a Python iterable or ndarray
// ---------------------------------------------------------------------------
template<typename T>
inline void assignCrystVector(CrystVector<T>& cv, nb::object obj)
{
    // Try ndarray first
    try {
        auto arr = nb::cast<nb_array_1d>(obj);
        assign_crystvec(cv, arr);
        return;
    } catch (...) {}
    // Fallback: iterate
    nb::list lst(obj);
    long n = static_cast<long>(nb::len(lst));
    cv.resize(n);
    for (long i = 0; i < n; ++i)
        cv(i) = static_cast<T>(nb::cast<double>(lst[i]));
}

// ---------------------------------------------------------------------------
// Helper: pointer container -> nb::list
// ---------------------------------------------------------------------------
template <class Container>
nb::list ptrcontainerToPyList(const Container& c)
{
    nb::list l;
    for (auto it = c.begin(); it != c.end(); ++it)
        l.append(nb::cast(*it, nb::rv_policy::reference));
    return l;
}

template <class Container>
nb::list containerToPyList(const Container& c)
{
    nb::list l;
    for (auto it = c.begin(); it != c.end(); ++it)
        l.append(nb::cast(*it));
    return l;
}

// ---------------------------------------------------------------------------
// pyIterableToSet — Python iterable -> std::set<T*>
// ---------------------------------------------------------------------------
template <class T>
std::set<T*> pyIterableToSet(nb::object obj)
{
    std::set<T*> result;
    for (nb::handle h : obj)
        result.insert(&nb::cast<T&>(h));
    return result;
}

// ---------------------------------------------------------------------------
// RefinableObj / ScatteringPower method-forwarding
//
// Scatterer, ScatteringPower, ScatteringData, and PowderPatternComponent all
// inherit RefinableObj VIRTUALLY in C++ (see extractRefinableObjArg above),
// and ScatteringPowerAtom/GlobalScatteringPower additionally inherit
// ScatteringPower virtually -- so nanobind cannot declare RefinableObj (or,
// for those last two, ScatteringPower) as their Python-visible base. Under
// Boost.Python, `bases<RefinableObj>` gave every one of these classes (and
// their concrete subclasses: Atom, Molecule, ZScatterer, ZPolyhedron,
// ScatteringPowerAtom, ScatteringPowerSphere, GlobalScatteringPower,
// DiffractionDataSingleCrystal, PowderPatternBackground,
// PowderPatternDiffraction) the full RefinableObj method surface for free,
// via ordinary Python inheritance, because Boost.Python's base registration
// performs a real cast rather than nanobind's cached fixed-offset scheme.
// nanobind can't do that, so today only a handful of these methods are
// reachable from Python on these classes (whatever a handful of prior,
// one-off `static_cast<RefinableObj&>(...)` shims happened to cover).
//
// bind_refinableobj_forwarding<T>()/bind_scatteringpower_forwarding<T>()
// restore the *rest* of that surface: one plain forwarding `.def()` per
// method, calling straight through T (implicit Derived&->Base& reference
// conversion, or a direct member call -- both resolved correctly by the
// compiler even across a virtual base, because T is fully known here at
// compile time; this is exactly the mechanism Boost.Python relied on, and
// is unrelated to nanobind's own generic, virtual-base-unsafe casting).
//
// Deliberately NOT the trampoline/NB_OVERRIDE mechanism: these become
// callable from Python (restoring Boost.Python-era behaviour), but a Python
// subclass overriding one of them still won't be seen by C++-internal
// virtual dispatch through a plain T&. That's a separate, harder problem
// (tracked for Crystal/PowderPattern/MonteCarloObj's existing UpdateDisplay
// overrides, which this file does not touch) and was explicitly out of
// scope for this pass.
// ---------------------------------------------------------------------------

inline nb_array_1d refobj_fwd_get_paramset(ObjCryst::RefinableObj& obj, unsigned long id)
{
    return crystvec_to_array(obj.GetParamSet(id));
}

inline void refobj_fwd_add_par(ObjCryst::RefinableObj& obj, ObjCryst::RefinablePar* p)
{
    obj.AddPar(p);
    obj.SetDeleteRefParInDestructor(0);
}

inline void refobj_fwd_add_par_obj(ObjCryst::RefinableObj& obj, nb::object o, const bool copyParam)
{
    obj.AddPar(extractRefinableObjArg(o), copyParam);
    obj.SetDeleteRefParInDestructor(0);
}

inline ObjCryst::RefinablePar& refobj_fwd_get_par_long(ObjCryst::RefinableObj& obj, const long i)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetPar(i);
}

inline ObjCryst::RefinablePar& refobj_fwd_get_par_string(ObjCryst::RefinableObj& obj, const std::string& s)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetPar(s);
}

inline ObjCryst::RefinablePar& refobj_fwd_get_par_not_fixed(ObjCryst::RefinableObj& obj, const long i)
{
    obj.SetDeleteRefParInDestructor(0);
    return obj.GetParNotFixed(i);
}

inline std::string refobj_fwd_xml(ObjCryst::RefinableObj& r)
{
    std::stringstream s;
    r.XMLOutput(s, 0);
    return s.str();
}

inline void refobj_fwd_xml_output(ObjCryst::RefinableObj& r, nb::object output, int indent)
{
    std::ostringstream os;
    os.precision(doublelim::digits10);
    r.XMLOutput(os, indent);
    output.attr("write")(os.str());
}

inline void refobj_fwd_xml_input(ObjCryst::RefinableObj& r, nb::object input, ObjCryst::XMLCrystTag& tag)
{
    std::string s = read_pyfile_to_string(input);
    std::istringstream is(s);
    r.XMLInput(is, tag);
}

inline void refobj_fwd_xml_input_string(ObjCryst::RefinableObj& r, const std::string& s)
{
    std::istringstream ss(s);
    ss.imbue(std::locale::classic());
    ObjCryst::XMLCrystTag tag;
    ss >> tag;
    r.XMLInput(ss, tag);
}

template <typename T, typename... Extra>
void bind_refinableobj_forwarding(nb::class_<T, Extra...>& cls)
{
    using ObjCryst::RefParType;
    using ObjCryst::RefinablePar;
    cls
        .def("PrepareForRefinement", &T::PrepareForRefinement)
        .def("FixAllPar",   &T::FixAllPar)
        .def("UnFixAllPar", &T::UnFixAllPar)
        .def("SetParIsFixed",
             nb::overload_cast<const long, const bool>(&T::SetParIsFixed))
        .def("SetParIsFixed",
             nb::overload_cast<const std::string&, const bool>(&T::SetParIsFixed))
        .def("SetParIsFixed",
             nb::overload_cast<const RefParType*, const bool>(&T::SetParIsFixed))
        .def("SetParIsUsed",
             nb::overload_cast<const std::string&, const bool>(&T::SetParIsUsed))
        .def("SetParIsUsed",
             nb::overload_cast<const RefParType*, const bool>(&T::SetParIsUsed))
        .def("GetNbPar",          &T::GetNbPar)
        .def("GetNbParNotFixed",  &T::GetNbParNotFixed)
        .def("GetPar",
             [](T& o, const long i) -> RefinablePar& { return refobj_fwd_get_par_long(o, i); },
             nb::rv_policy::reference_internal)
        .def("GetPar",
             [](T& o, const std::string& s) -> RefinablePar& { return refobj_fwd_get_par_string(o, s); },
             nb::rv_policy::reference_internal)
        .def("GetParNotFixed",
             [](T& o, const long i) -> RefinablePar& { return refobj_fwd_get_par_not_fixed(o, i); },
             nb::rv_policy::reference_internal)
        .def("AddPar",
             [](T& o, RefinablePar* p){ refobj_fwd_add_par(o, p); },
             nb::arg("par"), nb::keep_alive<1,2>())
        .def("AddPar",
             [](T& o, nb::object newRefParList, const bool copyParam){
                 refobj_fwd_add_par_obj(o, newRefParList, copyParam); },
             nb::arg("newRefParList"), nb::arg("copyParam") = false, nb::keep_alive<1,2>())
        .def("RemovePar", [](T& o, RefinablePar* p){ o.RemovePar(p); })
        .def("CreateParamSet", &T::CreateParamSet, nb::arg("name") = "")
        .def("ClearParamSet",  &T::ClearParamSet)
        .def("SaveParamSet",   &T::SaveParamSet)
        .def("RestoreParamSet",&T::RestoreParamSet)
        .def("GetParamSet",    [](T& o, unsigned long id){ return refobj_fwd_get_paramset(o, id); })
        .def("GetParamSet_ParNotFixedHumanValue", &T::GetParamSet_ParNotFixedHumanValue)
        .def("EraseAllParamSet", [](T& o){ o.EraseAllParamSet(); })
        .def("GetParamSetName",&T::GetParamSetName)
        .def("SetLimitsAbsolute",
             nb::overload_cast<const std::string&, const REAL, const REAL>(&T::SetLimitsAbsolute))
        .def("SetLimitsAbsolute",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(&T::SetLimitsAbsolute))
        .def("SetLimitsRelative",
             nb::overload_cast<const std::string&, const REAL, const REAL>(&T::SetLimitsRelative))
        .def("SetLimitsRelative",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(&T::SetLimitsRelative))
        .def("SetLimitsProportional",
             nb::overload_cast<const std::string&, const REAL, const REAL>(&T::SetLimitsProportional))
        .def("SetLimitsProportional",
             nb::overload_cast<const RefParType*, const REAL, const REAL>(&T::SetLimitsProportional))
        .def("SetGlobalOptimStep", &T::SetGlobalOptimStep)
        .def("GetSubObjRegistry",
             nb::overload_cast<>(&T::GetSubObjRegistry), nb::rv_policy::reference_internal)
        .def("GetClientRegistry",
             nb::overload_cast<>(&T::GetClientRegistry), nb::rv_policy::reference_internal)
        .def("GetOptionList", &T::GetOptionList, nb::rv_policy::reference)
        .def("GetNbOption",   &T::GetNbOption)
        .def("GetOption",
             nb::overload_cast<const unsigned int>(&T::GetOption), nb::rv_policy::reference_internal)
        .def("IsBeingRefined",          &T::IsBeingRefined)
        .def("BeginGlobalOptRandomMove",&T::BeginGlobalOptRandomMove)
        .def("ResetParList",            &T::ResetParList)
        .def("GetRefParListClock",      &T::GetRefParListClock, nb::rv_policy::reference_internal)
        .def("GetClockMaster",          &T::GetClockMaster,     nb::rv_policy::reference_internal)
        .def("AddRestraint",   &T::AddRestraint,  nb::keep_alive<1,2>())
        .def("RemoveRestraint",&T::RemoveRestraint)
        .def("GetClassName",  &T::GetClassName)
        .def("GetName",       &T::GetName)
        .def("SetName",       &T::SetName)
        .def("Print",         &T::Print)
        .def("RegisterClient",  &T::RegisterClient, nb::keep_alive<1,2>())
        .def("DeRegisterClient",&T::DeRegisterClient)
        .def("BeginOptimization",&T::BeginOptimization,
             nb::arg("allowApproximations") = false, nb::arg("enableRestraints") = false)
        .def("EndOptimization",       &T::EndOptimization)
        .def("RandomizeConfiguration",&T::RandomizeConfiguration)
        .def("GlobalOptRandomMove",   &T::GlobalOptRandomMove,
             nb::arg("mutationAmplitude"), nb::arg("type") = ObjCryst::gpRefParTypeObjCryst)
        .def("GetLogLikelihood",  &T::GetLogLikelihood)
        .def("GetNbLSQFunction",  &T::GetNbLSQFunction)
        .def("GetLSQCalc",   [](T& o, unsigned int i){ return crystvec_to_array(o.GetLSQCalc(i)); },
             nb::arg("idx"))
        .def("GetLSQObs",    [](T& o, unsigned int i){ return crystvec_to_array(o.GetLSQObs(i)); },
             nb::arg("idx"))
        .def("GetLSQWeight", [](T& o, unsigned int i){ return crystvec_to_array(o.GetLSQWeight(i)); },
             nb::arg("idx"))
        .def("GetLSQDeriv",  [](T& o, unsigned int i, RefinablePar& p){
                                 return crystvec_to_array(o.GetLSQDeriv(i, p)); },
             nb::arg("idx"), nb::arg("par"))
        .def("UpdateDisplay",     &T::UpdateDisplay)
        .def("GetRestraintCost",  &T::GetRestraintCost)
        .def("TagNewBestConfig",  &T::TagNewBestConfig)
        .def("XMLOutput",
             [](T& o, nb::object output, int indent){ refobj_fwd_xml_output(o, output, indent); },
             nb::arg("file"), nb::arg("indent") = 0)
        .def("xml", [](T& o){ return refobj_fwd_xml(o); })
        .def("XMLInput",
             [](T& o, nb::object input, ObjCryst::XMLCrystTag& tag){ refobj_fwd_xml_input(o, input, tag); },
             nb::arg("file"), nb::arg("tag"))
        .def("XMLInput",
             [](T& o, const std::string& s){ refobj_fwd_xml_input_string(o, s); },
             nb::arg("xml"))
        .def("GetGeneGroup",&T::GetGeneGroup)
        .def("int_ptr",    &T::int_ptr)
        ;
}

template <typename T, typename... Extra>
void bind_scatteringpower_forwarding(nb::class_<T, Extra...>& cls)
{
    cls
        .def("GetScatteringFactor",  &T::GetScatteringFactor,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("GetForwardScatteringFactor", &T::GetForwardScatteringFactor)
        .def("GetTemperatureFactor", &T::GetTemperatureFactor,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("GetResonantScattFactReal", &T::GetResonantScattFactReal,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("GetResonantScattFactImag", &T::GetResonantScattFactImag,
             nb::arg("data"), nb::arg("spgSymPosIndex") = -1)
        .def("IsScatteringFactorAnisotropic",  &T::IsScatteringFactorAnisotropic)
        .def("IsTemperatureFactorAnisotropic", &T::IsTemperatureFactorAnisotropic)
        .def("IsResonantScatteringAnisotropic",&T::IsResonantScatteringAnisotropic)
        .def("GetSymbol", &T::GetSymbol)
        .def("GetBiso",
             nb::overload_cast<>(&T::GetBiso, nb::const_))
        .def("SetBiso",  &T::SetBiso)
        .def("GetBij",
             nb::overload_cast<const size_t&, const size_t&>(&T::GetBij, nb::const_))
        .def("SetBij",
             nb::overload_cast<const size_t&, const size_t&, const REAL>(&T::SetBij))
        .def("IsIsotropic",          &T::IsIsotropic)
        .def("GetDynPopCorrIndex",   &T::GetDynPopCorrIndex)
        .def("GetNbScatteringPower", &T::GetNbScatteringPower)
        .def("GetLastChangeClock",   &T::GetLastChangeClock,
             nb::rv_policy::reference_internal)
        .def("GetRadius",                           &T::GetRadius)
        .def("GetMaximumLikelihoodPositionError",   &T::GetMaximumLikelihoodPositionError)
        .def("SetMaximumLikelihoodPositionError",   &T::SetMaximumLikelihoodPositionError)
        .def("GetMaximumLikelihoodNbGhostAtom",     &T::GetMaximumLikelihoodNbGhostAtom)
        .def("SetMaximumLikelihoodNbGhostAtom",     &T::SetMaximumLikelihoodNbGhostAtom)
        .def("GetMaximumLikelihoodParClock", &T::GetMaximumLikelihoodParClock,
             nb::rv_policy::reference_internal)
        .def("GetFormalCharge",  &T::GetFormalCharge)
        .def("SetFormalCharge",  &T::SetFormalCharge)
        .def("GetColourRGB", [](T& sp){ return nb::make_tuple(sp.GetColourRGB()[0],
                                                                sp.GetColourRGB()[1],
                                                                sp.GetColourRGB()[2]); })
        .def("GetColour",    [](T& sp){ return nb::make_tuple(sp.GetColourRGB()[0],
                                                                sp.GetColourRGB()[1],
                                                                sp.GetColourRGB()[2]); })
        .def("SetColour",
             nb::overload_cast<const float, const float, const float>(&T::SetColour),
             nb::arg("r"), nb::arg("g"), nb::arg("b"))
        // B11/B12/.../Biso properties: nb_scatteringpower.cpp exposes these
        // on ScatteringPower itself via file-local _GetBij<I,J>/_SetBij<I,J>
        // templates; those aren't visible here, so the properties are
        // reimplemented against T's own (non-virtual) GetBij/SetBij, which
        // resolve correctly across a virtual base the same way every other
        // forwarding .def() in this function does.
        .def_prop_rw("Biso",
             [](T& sp){ return sp.GetBiso(); },
             [](T& sp, const REAL b){ sp.SetBiso(b); })
        .def_prop_rw("B11",
             [](T& sp){ return sp.GetBij(1, 1); },
             [](T& sp, const REAL b){ sp.SetBij(1, 1, b); })
        .def_prop_rw("B22",
             [](T& sp){ return sp.GetBij(2, 2); },
             [](T& sp, const REAL b){ sp.SetBij(2, 2, b); })
        .def_prop_rw("B33",
             [](T& sp){ return sp.GetBij(3, 3); },
             [](T& sp, const REAL b){ sp.SetBij(3, 3, b); })
        .def_prop_rw("B12",
             [](T& sp){ return sp.GetBij(1, 2); },
             [](T& sp, const REAL b){ sp.SetBij(1, 2, b); })
        .def_prop_rw("B13",
             [](T& sp){ return sp.GetBij(1, 3); },
             [](T& sp, const REAL b){ sp.SetBij(1, 3, b); })
        .def_prop_rw("B23",
             [](T& sp){ return sp.GetBij(2, 3); },
             [](T& sp, const REAL b){ sp.SetBij(2, 3, b); })
        ;
}

#endif // HELPERS_NB_HPP
