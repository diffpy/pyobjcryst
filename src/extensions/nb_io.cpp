/*
 * pyobjcryst nanobind port — IO bindings (XMLCrystTag, XMLCrystFile*)
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>
#include <sstream>

#include <ObjCryst/RefinableObj/IO.h>
#include <ObjCryst/ObjCryst/IO.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

void _XMLCrystFileLoadAllObject(nb::object input, const bool verbose = false)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    if (verbose) {
        gag.release();
        muzzle.release();
    }
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    XMLCrystFileLoadAllObject(in);
}

void _XMLCrystFileSaveGlobal(nb::object output)
{
    std::ostringstream out;
    XMLCrystFileSaveGlobal(out);
    output.attr("write")(out.str());
}

} // namespace

void wrap_io(nb::module_& m)
{
    nb::class_<XMLCrystTag>(m, "XMLCrystTag")
        .def(nb::init<const std::string&, const bool, const bool>(),
             nb::arg("tagName"), nb::arg("isEndTag") = false, nb::arg("isEmptyTag") = false)
        .def("GetName",         &XMLCrystTag::GetName)
        .def("GetClassName",    &XMLCrystTag::GetClassName)
        .def("GetNbAttribute",  &XMLCrystTag::GetNbAttribute)
        .def("AddAttribute",    &XMLCrystTag::AddAttribute,
             nb::arg("attName"), nb::arg("attValue"))
        .def("GetAttribute",    &XMLCrystTag::GetAttribute,
             nb::arg("attNum"), nb::arg("attName"), nb::arg("attValue"))
        .def("GetAttributeName", &XMLCrystTag::GetAttributeName)
        .def("GetAttributeValue",&XMLCrystTag::GetAttributeValue)
        .def("SetIsEndTag",     &XMLCrystTag::SetIsEndTag)
        .def("IsEndTag",        &XMLCrystTag::IsEndTag)
        .def("SetIsEmptyTag",   &XMLCrystTag::SetIsEmptyTag)
        .def("IsEmptyTag",      &XMLCrystTag::IsEmptyTag)
        .def("Print",           &XMLCrystTag::Print)
        .def("__str__",         [](const XMLCrystTag& t){ return obj_str(t); })
        ;

    m.def("XMLCrystFileLoadAllObject", &_XMLCrystFileLoadAllObject,
          nb::arg("file"), nb::arg("verbose") = false);
    m.def("XMLCrystFileSaveGlobal", &_XMLCrystFileSaveGlobal,
          nb::arg("file"));
}
