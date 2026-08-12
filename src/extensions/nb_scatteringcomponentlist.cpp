/*
 * pyobjcryst nanobind port — ScatteringComponentList bindings
 */

#include <nanobind/nanobind.h>

#include <ObjCryst/ObjCryst/ScatteringPower.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

const ScatteringComponent& _getItem(const ScatteringComponentList& scl, long idx)
{
    long n = scl.GetNbComponent();
    if (idx < 0) idx += n;
    if (idx < 0 || idx >= n)
        throw nb::index_error("index out of range");
    return scl(idx);
}

bool _contains(const ScatteringComponentList& scl, const ScatteringComponent& sc)
{
    for (long i = 0; i < scl.GetNbComponent(); ++i)
        if (scl(i) == sc) return true;
    return false;
}

} // namespace

void wrap_scatteringcomponentlist(nb::module_& m)
{
    nb::class_<ScatteringComponentList>(m, "ScatteringComponentList")
        .def("Reset",          &ScatteringComponentList::Reset)
        .def("GetNbComponent", &ScatteringComponentList::GetNbComponent)
        .def("Print",          &ScatteringComponentList::Print)
        .def("__eq__",  [](const ScatteringComponentList& a, const ScatteringComponentList& b){ return a == b; })
        .def("__iadd__",[](ScatteringComponentList& a, const ScatteringComponentList& b) -> ScatteringComponentList& { a += b; return a; })
        .def("__iadd__",[](ScatteringComponentList& a, const ScatteringComponent& b) -> ScatteringComponentList& { a += b; return a; })
        .def("__str__", [](const ScatteringComponentList& s){ return obj_str(s); })
        .def("__len__", &ScatteringComponentList::GetNbComponent)
        .def("__getitem__", &_getItem, nb::rv_policy::reference_internal)
        .def("__contains__", &_contains)
        ;
}
