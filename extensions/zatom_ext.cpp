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
* boost::python bindings to ObjCryst::ZAtom.
*
* Changes from ObjCryst::ZAtom
* - XMLOutput and Input are not wrapped.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <ObjCryst/ObjCryst/ZScatterer.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

}

void wrap_zatom()
{

    /* This class is created internally by a ZScatterer, so it does not have an
     * init function
     */
    class_<ZAtom>("ZAtom", no_init)
        //init<ZScatterer&, const ScatteringPower*, const long, const double,
        //const long, const double, const long, const double, const double, const
        //string& >(
        //    (bp::arg("scatt"), bp::arg("pow"), bp::arg("atomBond")=0,
        //     bp::arg("bondLength")=1, bp::arg("atomAngle")=0,
        //     bp::arg("bondAngle")=0, bp::arg("atomDihedral")=0,
        //     bp::arg("popu")=1, bp::arg("name")="")
        //    )
        //[with_custodian_and_ward<2,1,with_custodian_and_ward<1,3> >()]
        //)
        // Methods
        .def("GetClassName", &ZAtom::GetClassName,
            return_value_policy<copy_const_reference>())
        .def("GetName", &ZAtom::GetName,
            return_value_policy<copy_const_reference>())
        .def("SetName", &ZAtom::SetName)
        .def("GetZScatterer", (ZScatterer& (ZAtom::*)()) &ZAtom::GetZScatterer,
            return_internal_reference<>())
        .def("GetZBondAtom", &ZAtom::GetZBondAtom)
        .def("GetZAngleAtom", &ZAtom::GetZAngleAtom)
        .def("GetZDihedralAngleAtom", &ZAtom::GetZDihedralAngleAtom)
        .def("GetZBondLength", &ZAtom::GetZBondLength,
            return_value_policy<copy_const_reference>())
        .def("GetZAngle", &ZAtom::GetZAngle,
            return_value_policy<copy_const_reference>())
        .def("GetZDihedralAngle", &ZAtom::GetZDihedralAngle,
            return_value_policy<copy_const_reference>())
        .def("GetOccupancy", &ZAtom::GetOccupancy,
            return_value_policy<copy_const_reference>())
        .def("GetScatteringPower", &ZAtom::GetScatteringPower,
            return_internal_reference<>())
        .def("SetZBondLength", &ZAtom::SetZBondLength)
        .def("SetZAngle", &ZAtom::SetZAngle)
        .def("SetZDihedralAngle", &ZAtom::SetZDihedralAngle)
        .def("SetOccupancy", &ZAtom::SetOccupancy)
        .def("SetScatteringPower", &ZAtom::SetScatteringPower,
            with_custodian_and_ward<1,2>())
        ;

}
