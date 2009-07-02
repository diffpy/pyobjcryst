/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::AsymmetricUnit.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/python/module.hpp>

void wrap_asymmetricunit();
void wrap_atom();
void wrap_crystal();
void wrap_general();
void wrap_globalscatteringpower();
void wrap_molatom();
void wrap_molbondangle();
void wrap_molbond();
void wrap_moldihedralangle();
void wrap_molecule();
void wrap_objregistry();
void wrap_polyhedron();
void wrap_quaternion();
void wrap_refinableobjclock();
void wrap_refinableobj();
void wrap_refinablepar();
void wrap_refobjopt();
void wrap_refpartype();
void wrap_registerconverters();
void wrap_restraint();
void wrap_rigidgroup();
void wrap_scatterer();
void wrap_scatteringcomponent();
void wrap_scatteringcomponentlist();
void wrap_scatteringpoweratom();
void wrap_scatteringpower();
void wrap_scatteringpowersphere();
void wrap_spacegroup();
void wrap_stretchmode();
void wrap_unitcell();
void wrap_zatom();
void wrap_zpolyhedron();
void wrap_zscatterer();

BOOST_PYTHON_MODULE(_pyobjcryst)
{
    // General functions and base classes
    wrap_registerconverters();
    wrap_general();
    wrap_objregistry();
    wrap_refinableobjclock();
    wrap_refpartype();
    wrap_restraint();
    wrap_refinablepar();
    wrap_refinableobj();

    // Scatter objects
    wrap_scatterer();

    wrap_atom();
    wrap_molecule();
    wrap_zscatterer();

    wrap_asymmetricunit();
    wrap_spacegroup();
    wrap_unitcell();
    wrap_crystal();

    wrap_scatteringpower();
    wrap_scatteringpoweratom();
    wrap_globalscatteringpower();
    wrap_scatteringpowersphere();

    wrap_molatom();
    wrap_molbondangle();
    wrap_molbond();
    wrap_moldihedralangle();
    
    wrap_polyhedron();
    wrap_quaternion();
    wrap_refobjopt();
    wrap_rigidgroup();
    wrap_scatteringcomponent();
    wrap_scatteringcomponentlist();
    wrap_stretchmode();
    wrap_zatom();
    wrap_zpolyhedron();
}

