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
* boost::python bindings to ObjCryst::ZPolyhedron.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/enum.hpp>

#include <string>

#include <ObjCryst/ObjCryst/ZScatterer.h>


namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


void wrap_zpolyhedron()
{

    class_<ZPolyhedron, bases<ZScatterer> >
        ("ZPolyhedron", init<const ZPolyhedron&>())
        /* Constructors */
        .def(init<const RegularPolyhedraType, Crystal&, double, double, double,
            const string&, const ScatteringPower*, const ScatteringPower*,
            double, double, double, double, double>(
            (bp::arg("type"), bp::arg("cryst"), bp::arg("x"),
             bp::arg("y"), bp::arg("z"), bp::arg("name"),
             bp::arg("centralAtomPow"), bp::arg("periphAtomPow"),
             bp::arg("centralPeriphDist"), bp::arg("ligandPopu")=1,
             bp::arg("phi")=0, bp::arg("chi")=0, bp::arg("psi")=0)
            )
            [with_custodian_and_ward<3,1,
                with_custodian_and_ward<1,8,
                    with_custodian_and_ward<1,9> > >()
            ]
        )
        ;

    enum_<RegularPolyhedraType>("RegularPolyhedraType")
        .value("TETRAHEDRON", TETRAHEDRON)
        .value("OCTAHEDRON", OCTAHEDRON)
        .value("SQUARE_PLANE", SQUARE_PLANE)
        .value("CUBE", CUBE)
        .value("ANTIPRISM_TETRAGONAL", ANTIPRISM_TETRAGONAL)
        .value("PRISM_TETRAGONAL_MONOCAP", PRISM_TETRAGONAL_MONOCAP)
        .value("PRISM_TETRAGONAL_DICAP ", PRISM_TETRAGONAL_DICAP)
        .value("PRISM_TRIGONAL ", PRISM_TRIGONAL)
        .value("PRISM_TRIGONAL_TRICAPPED", PRISM_TRIGONAL_TRICAPPED)
        .value("ICOSAHEDRON ", ICOSAHEDRON)
        .value("TRIANGLE_PLANE ", TRIANGLE_PLANE)
        ;
}
