/*
 * pyobjcryst nanobind port — UnitCell bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/UnitCell.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

nb::tuple FractionalToOrthonormal(const UnitCell& uc, double xd, double yd, double zd)
{
    REAL x=xd, y=yd, z=zd;
    uc.FractionalToOrthonormalCoords(x, y, z);
    return nb::make_tuple((double)x, (double)y, (double)z);
}

nb::tuple OrthonormalToFractional(const UnitCell& uc, double xd, double yd, double zd)
{
    REAL x=xd, y=yd, z=zd;
    uc.OrthonormalToFractionalCoords(x, y, z);
    return nb::make_tuple((double)x, (double)y, (double)z);
}

nb::tuple MillerToOrthonormal(const UnitCell& uc, double xd, double yd, double zd)
{
    REAL x=xd, y=yd, z=zd;
    uc.MillerToOrthonormalCoords(x, y, z);
    return nb::make_tuple((double)x, (double)y, (double)z);
}

nb::tuple OrthonormalToMiller(const UnitCell& uc, double xd, double yd, double zd)
{
    REAL x=xd, y=yd, z=zd;
    uc.OrthonormalToMillerCoords(x, y, z);
    return nb::make_tuple((double)x, (double)y, (double)z);
}

void _seta(UnitCell& u, double v) { u.GetPar("a").SetValue(v); }
double _geta(UnitCell& u) { return u.GetLatticePar(0); }
void _setb(UnitCell& u, double v) { u.GetPar("b").SetValue(v); }
double _getb(UnitCell& u) { return u.GetLatticePar(1); }
void _setc(UnitCell& u, double v) { u.GetPar("c").SetValue(v); }
double _getc(UnitCell& u) { return u.GetLatticePar(2); }

void _setalpha(UnitCell& u, double v) {
    if (v <= 0 || v >= M_PI) throw ObjCrystException("alpha must be within ]0;pi[");
    RefinablePar& p = u.GetPar("alpha");
    if (p.IsUsed()) p.SetValue(v);
}
double _getalpha(UnitCell& u) { return u.GetLatticePar(3); }

void _setbeta(UnitCell& u, double v) {
    if (v <= 0 || v >= M_PI) throw ObjCrystException("beta must be within ]0;pi[");
    RefinablePar& p = u.GetPar("beta");
    if (p.IsUsed()) p.SetValue(v);
}
double _getbeta(UnitCell& u) { return u.GetLatticePar(4); }

void _setgamma(UnitCell& u, double v) {
    if (v <= 0 || v >= M_PI) throw ObjCrystException("gamma must be within ]0;pi[");
    RefinablePar& p = u.GetPar("gamma");
    if (p.IsUsed()) p.SetValue(v);
}
double _getgamma(UnitCell& u) { return u.GetLatticePar(5); }

void SafeChangeSpaceGroup(UnitCell& u, const std::string& sgid)
{
    MuteObjCrystUserInfo muzzle;
    u.ChangeSpaceGroup(sgid);
}

} // namespace

void wrap_unitcell(nb::module_& m)
{
    m.attr("refpartype_unitcell")        = gpRefParTypeUnitCell;
    m.attr("refpartype_unitcell_length") = gpRefParTypeUnitCellLength;
    m.attr("refpartype_unitcell_angle")  = gpRefParTypeUnitCellAngle;

    nb::class_<UnitCell, RefinableObj>(m, "UnitCell")
        .def(nb::init<>())
        .def(nb::init<const double, const REAL, const double, const std::string&>())
        .def(nb::init<const double, const REAL, const double,
                      const double, const double, const double,
                      const std::string&>())
        .def(nb::init<const UnitCell&>())
        .def("GetLatticePar",
             nb::overload_cast<>(&UnitCell::GetLatticePar, nb::const_))
        .def("GetLatticePar",
             nb::overload_cast<const int>(&UnitCell::GetLatticePar, nb::const_))
        .def("GetClockLatticePar",   &UnitCell::GetClockLatticePar,
             nb::rv_policy::reference_internal)
        .def("GetBMatrix",     [](UnitCell& u){ return crystmat_to_array(u.GetBMatrix()); })
        .def("GetOrthMatrix",  [](UnitCell& u){ return crystmat_to_array(u.GetOrthMatrix()); })
        .def("GetClockMetricMatrix", &UnitCell::GetClockMetricMatrix,
             nb::rv_policy::reference_internal)
        .def("GetOrthonormalCoords", &UnitCell::GetOrthonormalCoords)
        .def("FractionalToOrthonormalCoords", &FractionalToOrthonormal)
        .def("OrthonormalToFractionalCoords", &OrthonormalToFractional)
        .def("MillerToOrthonormalCoords",     &MillerToOrthonormal)
        .def("OrthonormalToMillerCoords",     &OrthonormalToMiller)
        .def("GetSpaceGroup",
             nb::overload_cast<>(&UnitCell::GetSpaceGroup),
             nb::rv_policy::reference_internal)
        .def("ChangeSpaceGroup", &SafeChangeSpaceGroup)
        .def("GetVolume",        &UnitCell::GetVolume)
        .def("__str__",  [](const UnitCell& u){ return obj_str(u); })
        .def_prop_rw("a",     &_geta,     &_seta)
        .def_prop_rw("b",     &_getb,     &_setb)
        .def_prop_rw("c",     &_getc,     &_setc)
        .def_prop_rw("alpha", &_getalpha, &_setalpha)
        .def_prop_rw("beta",  &_getbeta,  &_setbeta)
        .def_prop_rw("gamma", &_getgamma, &_setgamma)
        ;
}
