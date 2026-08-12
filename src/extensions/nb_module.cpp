/*
 * pyobjcryst nanobind port — module entry point (_pyobjcryst)
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#undef B0
#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace nb = nanobind;

// Forward declarations for all wrap_* functions
void wrap_general(nb::module_&);
void wrap_io(nb::module_&);
void wrap_objregistry(nb::module_&);
void wrap_quaternion(nb::module_&);
void wrap_refinableobjclock(nb::module_&);
void wrap_refobjopt(nb::module_&);
void wrap_refpartype(nb::module_&);
void wrap_restraint(nb::module_&);
void wrap_refinablepar(nb::module_&);
void wrap_refinableobj(nb::module_&);
void wrap_scatteringdata(nb::module_&);
void wrap_scatterer(nb::module_&);
void wrap_scatteringpower(nb::module_&);
void wrap_zscatterer(nb::module_&);
void wrap_unitcell(nb::module_&);
void wrap_powderpatterncomponent(nb::module_&);
void wrap_reflectionprofile(nb::module_&);
void wrap_asymmetricunit(nb::module_&);
void wrap_atom(nb::module_&);
void wrap_crystal(nb::module_&);
void wrap_diffractiondatasinglecrystal(nb::module_&);
void wrap_globaloptim(nb::module_&);
void wrap_globalscatteringpower(nb::module_&);
void wrap_indexing(nb::module_&);
void wrap_lsq(nb::module_&);
void wrap_molatom(nb::module_&);
void wrap_molbond(nb::module_&);
void wrap_molbondangle(nb::module_&);
void wrap_moldihedralangle(nb::module_&);
void wrap_molecule(nb::module_&);
void wrap_polyhedron(nb::module_&);
void wrap_powderpattern(nb::module_&);
void wrap_powderpatternbackground(nb::module_&);
void wrap_powderpatterndiffraction(nb::module_&);
void wrap_radiation(nb::module_&);
void wrap_rigidgroup(nb::module_&);
void wrap_scatteringcomponent(nb::module_&);
void wrap_scatteringcomponentlist(nb::module_&);
void wrap_scatteringpoweratom(nb::module_&);
void wrap_scatteringpowersphere(nb::module_&);
void wrap_spacegroup(nb::module_&);
void wrap_stretchmode(nb::module_&);
void wrap_zatom(nb::module_&);
void wrap_zpolyhedron(nb::module_&);

NB_MODULE(_pyobjcryst, m)
{
    m.doc() = "pyobjcryst — Python bindings to ObjCryst++ (nanobind port)";

    // Create ObjCrystException as a Python exception class
    PyObject* exc_class = PyErr_NewException("_pyobjcryst.ObjCrystException", nullptr, nullptr);
    m.attr("ObjCrystException") = nb::steal<nb::object>(exc_class);

    // Register translator so C++ ObjCrystException -> Python ObjCrystException
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* exc_ptr) {
            try {
                std::rethrow_exception(p);
            } catch (const ObjCryst::ObjCrystException& e) {
                PyErr_SetString(reinterpret_cast<PyObject*>(exc_ptr), e.message.c_str());
            }
        }, exc_class
    );

    // General enums, version
    wrap_general(m);

    // I/O helpers (XMLCrystTag etc.)
    wrap_io(m);

    // ObjRegistry<T> template instantiations
    wrap_objregistry(m);

    // Simple value types
    wrap_quaternion(m);
    wrap_refinableobjclock(m);
    wrap_refobjopt(m);
    wrap_refpartype(m);

    // Core refinable hierarchy (order matters: base before derived)
    wrap_restraint(m);
    wrap_refinablepar(m);
    wrap_refinableobj(m);

    // Scattering base classes
    wrap_scatteringdata(m);
    wrap_scatterer(m);
    wrap_scatteringpower(m);
    wrap_zscatterer(m);
    wrap_unitcell(m);
    wrap_powderpatterncomponent(m);
    wrap_reflectionprofile(m);

    // Concrete classes
    wrap_asymmetricunit(m);
    wrap_spacegroup(m);
    wrap_scatteringcomponent(m);
    wrap_scatteringcomponentlist(m);
    wrap_scatteringpoweratom(m);
    wrap_scatteringpowersphere(m);
    wrap_atom(m);
    wrap_crystal(m);
    wrap_diffractiondatasinglecrystal(m);
    wrap_globaloptim(m);
    wrap_globalscatteringpower(m);
    wrap_indexing(m);
    wrap_lsq(m);
    wrap_rigidgroup(m);
    wrap_molatom(m);
    wrap_molbond(m);
    wrap_molbondangle(m);
    wrap_moldihedralangle(m);
    wrap_stretchmode(m);
    wrap_molecule(m);
    wrap_polyhedron(m);
    wrap_powderpattern(m);
    wrap_powderpatternbackground(m);
    wrap_powderpatterndiffraction(m);
    wrap_radiation(m);
    wrap_zatom(m);
    wrap_zpolyhedron(m);
}
