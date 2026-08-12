/*
 * pyobjcryst nanobind port — SpaceGroup bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <ObjCryst/ObjCryst/SpaceGroup.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

nb::list GetTranslationVectors(const SpaceGroup& sg)
{
    const std::vector<SpaceGroup::TRx>& tv = sg.GetTranslationVectors();
    nb::list outlist;
    for (auto& vec : tv) {
        CrystVector_REAL translation(3);
        for (int i = 0; i < 3; ++i) translation(i) = vec.tr[i];
        outlist.append(crystvec_to_array(translation));
    }
    return outlist;
}

nb::list GetSymmetryOperations(const SpaceGroup& sg)
{
    const std::vector<SpaceGroup::SMx>& sv = sg.GetSymmetryOperations();
    nb::list outlist;
    for (auto& tup : sv) {
        CrystVector_REAL translation(3);
        for (int i = 0; i < 3; ++i) translation(i) = tup.tr[i];
        CrystMatrix_REAL rotation(3, 3);
        for (int idx = 0; idx < 9; ++idx) rotation(idx/3, idx%3) = tup.mx[idx];
        outlist.append(nb::make_tuple(crystvec_to_array(translation), crystmat_to_array(rotation)));
    }
    return outlist;
}

SpaceGroup* CreateSpaceGroup(const std::string& sgid)
{
    MuteObjCrystUserInfo muzzle;
    return new SpaceGroup(sgid);
}

void SafeChangeSpaceGroup(SpaceGroup& sg, const std::string& sgid)
{
    MuteObjCrystUserInfo muzzle;
    sg.ChangeSpaceGroup(sgid);
}

} // namespace

void wrap_spacegroup(nb::module_& m)
{
    nb::class_<SpaceGroup>(m, "SpaceGroup")
        .def(nb::init<>())
        .def("__init__",
             [](SpaceGroup* sg, const std::string& sgid) {
                 MuteObjCrystUserInfo muzzle;
                 new (sg) SpaceGroup(sgid);
             }, nb::arg("spacegroup") = "P1")
        .def("ChangeSpaceGroup",          &SafeChangeSpaceGroup)
        .def("GetName",                   &SpaceGroup::GetName)
        .def("IsInAsymmetricUnit",        &SpaceGroup::IsInAsymmetricUnit)
        .def("ChangeToAsymmetricUnit",    &SpaceGroup::ChangeToAsymmetricUnit)
        .def("GetAsymUnit",               &SpaceGroup::GetAsymUnit,
             nb::rv_policy::reference_internal)
        .def("GetSpaceGroupNumber",       &SpaceGroup::GetSpaceGroupNumber)
        .def("IsCentrosymmetric",         &SpaceGroup::IsCentrosymmetric)
        .def("GetNbTranslationVectors",   &SpaceGroup::GetNbTranslationVectors)
        .def("GetTranslationVectors",     &GetTranslationVectors)
        .def("GetSymmetryOperations",     &GetSymmetryOperations)
        .def("GetAllSymmetrics",
             [](const SpaceGroup& sg, REAL x, REAL y, REAL z,
                bool noCenter, bool noTransl, bool noIdentical) {
                 return crystmat_to_array(sg.GetAllSymmetrics(x, y, z, noCenter, noTransl, noIdentical));
             },
             nb::arg("x"), nb::arg("y"), nb::arg("z"),
             nb::arg("noCenter") = false, nb::arg("noTransl") = false,
             nb::arg("noIdentical") = false)
        .def("GetNbSymmetrics",           &SpaceGroup::GetNbSymmetrics,
             nb::arg("noCenter") = false, nb::arg("noTransl") = false)
        .def("GetInversionCenter", [](const SpaceGroup& sg) {
                 return crystvec_to_array(sg.GetInversionCenter()); })
        .def("Print",                     &SpaceGroup::Print)
        .def("HasInversionCenter",        &SpaceGroup::HasInversionCenter)
        .def("IsInversionCenterAtOrigin", &SpaceGroup::IsInversionCenterAtOrigin)
        .def("GetClockSpaceGroup",        &SpaceGroup::GetClockSpaceGroup,
             nb::rv_policy::reference_internal)
        .def("GetUniqueAxis",             &SpaceGroup::GetUniqueAxis)
        .def("GetExtension",              &SpaceGroup::GetExtension)
        .def("GetAllEquivRefl",           [](const SpaceGroup& sg, REAL h, REAL k, REAL l,
                                             bool excFriedel, bool forceFriedel) {
                 return crystmat_to_array(sg.GetAllEquivRefl(h, k, l, excFriedel, forceFriedel));
             },
             nb::arg("h"), nb::arg("k"), nb::arg("l"),
             nb::arg("excludeFriedelMate") = false,
             nb::arg("forceFriedelLaw") = false)
        .def("IsReflSystematicAbsent",    &SpaceGroup::IsReflSystematicAbsent)
        .def("IsReflCentric",             &SpaceGroup::IsReflCentric)
        .def("GetExpectedIntensityFactor",&SpaceGroup::GetExpectedIntensityFactor)
        .def("__str__",  &SpaceGroup::GetName)
        .def("__repr__", &SpaceGroup::GetName)
        ;
}
