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
* boost::python bindings to ObjCryst::ZScatterer.
*
* Changes from ObjCryst++
* - XMLOutput and Input are not wrapped.
*
* - Import and output is not implemented yet.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/ObjCryst/ZScatterer.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

void wrap_zscatterer()
{

    class_<ZScatterer, bases<Scatterer> >
        ("ZScatterer", init<const ZScatterer&>(bp::arg("old")))
        /* Constructors */
        .def(init<const string&, Crystal&, double, double, double, double, double,
            double>
            ((bp::arg("name"), bp::arg("cryst"), bp::arg("x")=0, bp::arg("y")=0,
              bp::arg("z")=0, bp::arg("phi")=0, bp::arg("chi")=0,
              bp::arg("psi")=0)
            ) [with_custodian_and_ward<1,6>()])
        /* Methods */
        .def("GetClassName", &ZAtom::GetClassName,
            return_value_policy<copy_const_reference>())
        .def("AddAtom", &ZScatterer::AddAtom,
            (bp::arg("name"), bp::arg("pow"),
             bp::arg("atomBond"), bp::arg("bondLength"),
             bp::arg("atomAngle"), bp::arg("bondAngle"),
             bp::arg("atomDihedral"), bp::arg("dihedralAngle"),
             bp::arg("popu")=1.0
             ), with_custodian_and_ward<1,3>()
            )
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
        .def("GetZAtomRegistry", &ZScatterer::GetZAtomRegistry,
            return_value_policy<copy_const_reference>())
        .def("GetXCoord", &ZScatterer::GetXCoord,
            return_value_policy<copy_const_reference>())
        .def("GetYCoord", &ZScatterer::GetYCoord,
            return_value_policy<copy_const_reference>())
        .def("GetZCoord", &ZScatterer::GetZCoord,
            return_value_policy<copy_const_reference>())
        .def("SetCenterAtomIndex", &ZScatterer::SetCenterAtomIndex)
        .def("GetCenterAtomIndex", &ZScatterer::GetCenterAtomIndex)
        // Python-only methods
        .def("__str__", &__str__<ZScatterer>)
        ;

}
