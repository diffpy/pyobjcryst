/*
 * pyobjcryst nanobind port — ObjRegistry<T> bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/make_iterator.h>

#undef B0
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/Scatterer.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

template <class T>
nb::list _getObjSlice(ObjRegistry<T>& reg, nb::slice s)
{
    nb::list l;
    for (auto it = reg.begin(); it != reg.end(); ++it)
        l.append(nb::cast(*it, nb::rv_policy::reference));
    // Apply slice using Python
    return nb::cast<nb::list>(nb::object(l).attr("__getitem__")(s));
}

template <class T>
MonteCarloObj& _getObjCastMonteCarlo(ObjRegistry<T>& reg, const unsigned int i)
{
    T* obj = &reg.GetObj(i);
    MonteCarloObj* p = dynamic_cast<MonteCarloObj*>(obj);
    if (!p) throw ObjCryst::ObjCrystException("Cannot cast to MonteCarloObj");
    return *p;
}

template <>
MonteCarloObj& _getObjCastMonteCarlo<ZAtom>(ObjRegistry<ZAtom>& reg, const unsigned int i)
{
    throw ObjCryst::ObjCrystException("Cannot cast a ZAtom to a MonteCarloObj");
}

template <class T>
void wrapObjRegistry(nb::class_<ObjRegistry<T>>& c)
{
    c
    .def("Register", &ObjRegistry<T>::Register)
    .def("DeRegister",
         nb::overload_cast<T&>(&ObjRegistry<T>::DeRegister))
    .def("DeRegister",
         nb::overload_cast<const std::string&>(&ObjRegistry<T>::DeRegister))
    .def("DeRegisterAll", &ObjRegistry<T>::DeRegisterAll)
    .def("GetObj",
         nb::overload_cast<const unsigned int>(&ObjRegistry<T>::GetObj),
         nb::rv_policy::reference_internal)
    .def("GetObj",
         nb::overload_cast<const std::string&>(&ObjRegistry<T>::GetObj),
         nb::rv_policy::reference_internal)
    .def("GetObj",
         nb::overload_cast<const std::string&, const std::string&>(&ObjRegistry<T>::GetObj),
         nb::rv_policy::reference_internal)
    .def("GetNb",   &ObjRegistry<T>::GetNb)
    .def("Print",   &ObjRegistry<T>::Print)
    .def("SetName", &ObjRegistry<T>::SetName)
    .def("GetName", &ObjRegistry<T>::GetName)
    .def("Find",
         nb::overload_cast<const std::string&>(&ObjRegistry<T>::Find, nb::const_))
    .def("Find",
         nb::overload_cast<const std::string&, const std::string&, const bool>(
             &ObjRegistry<T>::Find, nb::const_))
    .def("Find",
         nb::overload_cast<const T&>(&ObjRegistry<T>::Find, nb::const_))
    .def("GetRegistryClock", &ObjRegistry<T>::GetRegistryClock,
         nb::rv_policy::reference_internal)
    .def("__str__", [](ObjRegistry<T>& r){ return obj_str(r); })
    .def("__len__", &ObjRegistry<T>::GetNb)
    .def("__getitem__",
         nb::overload_cast<const unsigned int>(&ObjRegistry<T>::GetObj),
         nb::rv_policy::reference_internal)
    .def("__getitem__", &_getObjSlice<T>)
    .def("__iter__",
         [](ObjRegistry<T>& r) {
             return nb::make_iterator(nb::find(nb::type<ObjRegistry<T>>()), "iterator",
                                      r.begin(), r.end());
         }, nb::keep_alive<0, 1>())
    .def("_getObjCastMonteCarlo", &_getObjCastMonteCarlo<T>,
         nb::rv_policy::reference_internal)
    ;
}

} // namespace

void wrap_objregistry(nb::module_& m)
{
    {
        nb::class_<ObjRegistry<Crystal>> c(m, "CrystalRegistry", nb::dynamic_attr());
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<Crystal>(c);
    }
    {
        nb::class_<ObjRegistry<PowderPattern>> c(m, "PowderPatternRegistry", nb::dynamic_attr());
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<PowderPattern>(c);
    }
    {
        nb::class_<ObjRegistry<DiffractionDataSingleCrystal>> c(m, "DiffractionDataSingleCrystalRegistry", nb::dynamic_attr());
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<DiffractionDataSingleCrystal>(c);
    }
    {
        nb::class_<ObjRegistry<OptimizationObj>> c(m, "OptimizationObjRegistry", nb::dynamic_attr());
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<OptimizationObj>(c);
    }
    {
        nb::class_<ObjRegistry<RefinableObj>> c(m, "RefinableObjRegistry", nb::dynamic_attr());
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<RefinableObj>(c);
    }
    {
        nb::class_<ObjRegistry<RefObjOpt>> c(m, "RefObjOptRegistry", nb::dynamic_attr());
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<RefObjOpt>(c);
    }
    {
        nb::class_<ObjRegistry<Scatterer>> c(m, "ScattererRegistry");
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<Scatterer>(c);
    }
    {
        nb::class_<ObjRegistry<ScatteringPower>> c(m, "ScatteringPowerRegistry");
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<ScatteringPower>(c);
    }
    {
        nb::class_<ObjRegistry<ScatteringPowerAtom>> c(m, "ScatteringPowerAtomRegistry");
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<ScatteringPowerAtom>(c);
    }
    {
        nb::class_<ObjRegistry<ZAtom>> c(m, "ZAtomRegistry");
        c.def(nb::init<const std::string&>());
        wrapObjRegistry<ZAtom>(c);
    }
}
