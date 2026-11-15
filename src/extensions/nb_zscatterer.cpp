#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>
#include <sstream>
#include <ObjCryst/ObjCryst/ZScatterer.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyZScatterer : ZScatterer {
    NB_TRAMPOLINE(ZScatterer, 1);
    // Explicit constructors to help nanobind find them
    PyZScatterer(const std::string& name, Crystal& cryst,
                 REAL x=0, REAL y=0, REAL z=0,
                 REAL phi=0, REAL chi=0, REAL psi=0)
        : ZScatterer(name, cryst, x, y, z, phi, chi, psi) {}
    PyZScatterer(const ZScatterer& old) : ZScatterer(old) {}
    ZScatterer* CreateCopy() const override { NB_OVERRIDE_PURE(CreateCopy); }
};

namespace {
void _ImportFenskeHallZMatrix(ZScatterer& scatt, nb::object input, bool named)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    scatt.ImportFenskeHallZMatrix(in, named);
}
} // namespace

void wrap_zscatterer(nb::module_& m)
{
    nb::class_<ZScatterer, Scatterer, PyZScatterer>(m, "ZScatterer")
        .def(nb::init<const ZScatterer&>(), nb::arg("old"))
        .def(nb::init<const std::string&, Crystal&, REAL, REAL, REAL,
                      REAL, REAL, REAL>(),
             nb::arg("name"), nb::arg("cryst"),
             nb::arg("x") = 0., nb::arg("y") = 0., nb::arg("z") = 0.,
             nb::arg("phi") = 0., nb::arg("chi") = 0., nb::arg("psi") = 0.,
             nb::keep_alive<1,3>())
        .def("GetClassName", &ZScatterer::GetClassName)
        .def("AddAtom", &ZScatterer::AddAtom,
             nb::arg("name"), nb::arg("pow"),
             nb::arg("atomBond"), nb::arg("bondLength"),
             nb::arg("atomAngle"), nb::arg("bondAngle"),
             nb::arg("atomDihedral"), nb::arg("dihedralAngle"),
             nb::arg("popu") = 1.0,
             nb::keep_alive<1,3>())
        .def("GetPhi", &ZScatterer::GetPhi)
        .def("GetChi", &ZScatterer::GetChi)
        .def("GetPsi", &ZScatterer::GetPsi)
        .def("SetPhi", &ZScatterer::SetPhi)
        .def("SetChi", &ZScatterer::SetChi)
        .def("SetPsi", &ZScatterer::SetPsi)
        .def("GetZAtomX", &ZScatterer::GetZAtomX)
        .def("GetZAtomY", &ZScatterer::GetZAtomY)
        .def("GetZAtomZ", &ZScatterer::GetZAtomZ)
        .def("GetZBondAtom", &ZScatterer::GetZBondAtom)
        .def("GetZAngleAtom", &ZScatterer::GetZAngleAtom)
        .def("GetZDihedralAngleAtom", &ZScatterer::GetZDihedralAngleAtom)
        .def("GetZBondLength", &ZScatterer::GetZBondLength)
        .def("GetZAngle", &ZScatterer::GetZAngle)
        .def("GetZDihedralAngle", &ZScatterer::GetZDihedralAngle)
        .def("SetZBondLength", &ZScatterer::SetZBondLength)
        .def("SetZAngle", &ZScatterer::SetZAngle)
        .def("SetZDihedralAngle", &ZScatterer::SetZDihedralAngle)
        .def("GetZAtomRegistry",
             [](ZScatterer& z) -> const ObjRegistry<ZAtom>& { return z.GetZAtomRegistry(); },
             nb::rv_policy::reference_internal)
        .def("GetXCoord", [](ZScatterer& z){ return crystvec_to_array(z.GetXCoord()); })
        .def("GetYCoord", [](ZScatterer& z){ return crystvec_to_array(z.GetYCoord()); })
        .def("GetZCoord", [](ZScatterer& z){ return crystvec_to_array(z.GetZCoord()); })
        .def("SetCenterAtomIndex", &ZScatterer::SetCenterAtomIndex)
        .def("GetCenterAtomIndex", &ZScatterer::GetCenterAtomIndex)
        .def("ImportFenskeHallZMatrix", &_ImportFenskeHallZMatrix,
             nb::arg("input"), nb::arg("named") = false)
        .def("__str__", [](ZScatterer& z){ return obj_str(z); })
        ;
}
