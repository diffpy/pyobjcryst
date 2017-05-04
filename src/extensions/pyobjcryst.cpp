/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* definition of the _pyobjcryst Python extension module.
*
*****************************************************************************/

#include <boost/python/module.hpp>

// Initialize numpy here.
#include "pyobjcryst_numpy_setup.hpp"
#include <numpy/arrayobject.h>


void wrap_asymmetricunit();
void wrap_atom();
void wrap_crystal();
void wrap_diffractiondatasinglecrystal();
void wrap_general();
void wrap_globalscatteringpower();
void wrap_io();
void wrap_molatom();
void wrap_molbond();
void wrap_molbondangle();
void wrap_moldihedralangle();
void wrap_molecule();
void wrap_objregistry();
void wrap_polyhedron();
void wrap_powderpattern();
void wrap_powderpatternbackground();
void wrap_powderpatterncomponent();
void wrap_powderpatterndiffraction();
void wrap_quaternion();
void wrap_refinableobj();
void wrap_refinableobjclock();
void wrap_refinablepar();
void wrap_reflectionprofile();
void wrap_refobjopt();
void wrap_refpartype();
void wrap_registerconverters();
void wrap_restraint();
void wrap_rigidgroup();
void wrap_scatterer();
void wrap_scatteringcomponent();
void wrap_scatteringcomponentlist();
void wrap_scatteringdata();
void wrap_scatteringpower();
void wrap_scatteringpoweratom();
void wrap_scatteringpowersphere();
void wrap_spacegroup();
void wrap_stretchmode();
void wrap_unitcell();
void wrap_zatom();
void wrap_zpolyhedron();
void wrap_zscatterer();

namespace {

#if PY_MAJOR_VERSION >= 3
    void* initialize_numpy() { import_array(); return NULL; }
#else
    void initialize_numpy() { import_array(); }
#endif

}   // namespace

// Wrappers must be called according to inheritance hierarchy
BOOST_PYTHON_MODULE(_pyobjcryst)
{
    // initialize numpy module
    initialize_numpy();

    // General stuff
    wrap_general();
    wrap_io();
    wrap_objregistry();
    wrap_quaternion();
    wrap_refinableobjclock();
    wrap_refobjopt();
    wrap_refpartype();
    wrap_registerconverters();

    // Core objects
    wrap_restraint();
    wrap_refinablepar();
    wrap_refinableobj();

    // Other base classes
    wrap_scatteringdata();
    wrap_scatterer();
    wrap_scatteringpower();
    wrap_zscatterer();
    wrap_unitcell();
    wrap_powderpatterncomponent();
    wrap_reflectionprofile();

    // Other stuff in no particular order.
    wrap_asymmetricunit();
    wrap_atom();
    wrap_crystal();
    wrap_diffractiondatasinglecrystal();
    wrap_globalscatteringpower();
    wrap_molatom();
    wrap_molbond();
    wrap_molbondangle();
    wrap_moldihedralangle();
    wrap_molecule();
    wrap_polyhedron();
    wrap_powderpattern();
    wrap_powderpatternbackground();
    wrap_powderpatterndiffraction();
    wrap_rigidgroup();
    wrap_scatteringcomponent();
    wrap_scatteringcomponentlist();
    wrap_scatteringpoweratom();
    wrap_scatteringpowersphere();
    wrap_spacegroup();
    wrap_stretchmode();
    wrap_zatom();
    wrap_zpolyhedron();
}
