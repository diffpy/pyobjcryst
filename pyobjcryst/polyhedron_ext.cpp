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
* boost::python bindings to ObjCryst::Polyhedron module.  
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Polyhedron.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/make_constructor.hpp>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

// FIXME - put the argument names in

namespace {

} // namespace


BOOST_PYTHON_MODULE(_polyhedron)
{

    def("MakeTetrahedron", &MakeTetrahedron,
         return_value_policy<manage_new_object>());

    def("MakeOctahedron", &MakeOctahedron,
        return_value_policy<manage_new_object>());

    def("MakeSquarePlane", &MakeSquarePlane,
        return_value_policy<manage_new_object>());

    def("MakeCube", &MakeCube,
        return_value_policy<manage_new_object>());

    def("MakeAntiPrismTetragonal", &MakeAntiPrismTetragonal,
        return_value_policy<manage_new_object>());

    def("MakePrismTrigonal", &MakePrismTrigonal,
        return_value_policy<manage_new_object>());

    def("MakeIcosahedron", &MakeIcosahedron,
        return_value_policy<manage_new_object>());

    def("MakeTriangle", &MakeTriangle,
        return_value_policy<manage_new_object>());

}
