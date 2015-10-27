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
* boost::python bindings to ObjCryst::Quaternion.
*
* Changes from ObjCryst::Quaternion
* - IO is not wrapped
* - Q0, Q1, Q2 and Q3 are wrapped as properties, rather than functions.
* - RotateVector overloaded to return tuple of the mutated arguments.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/tuple.hpp>

#include <ObjCryst/ObjCryst/Molecule.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

double getQ0(const Quaternion& q)
{
    return q.Q0();
}

double getQ1(const Quaternion& q)
{
    return q.Q1();
}

double getQ2(const Quaternion& q)
{
    return q.Q2();
}

double getQ3(const Quaternion& q)
{
    return q.Q3();
}

void setQ0(Quaternion& q, double val)
{
    q.Q0() = val;
}

void setQ1(Quaternion& q, double val)
{
    q.Q1() = val;
}

void setQ2(Quaternion& q, double val)
{
    q.Q2() = val;
}

void setQ3(Quaternion& q, double val)
{
    q.Q3() = val;
}

// Overloaded to return a tuple
bp::tuple _RotateVector( const Quaternion& q, double v1, double v2, double v3 )
{

    q.RotateVector(v1, v2, v3);
    return bp::make_tuple(v1, v2, v3);

}

} // namespace

void wrap_quaternion()
{

    class_<Quaternion>("Quaternion")
        .def(init<const double, const double, const double, const double, bool>(
            (bp::arg("q0"), bp::arg("q1"), bp::arg("q2"), bp::arg("q3"),
             bp::arg("unit")=true
            ))
            )
        .def("GetConjugate", &Quaternion::GetConjugate)
        //.def("RotateVector", &Quaternion::RotateVector,
        //    (bp::arg("v1"), bp::arg("v2"), bp::arg("v3")))
        .def("RotateVector", &_RotateVector,
            (bp::arg("v1"), bp::arg("v2"), bp::arg("v3")))
        .def("Normalize", &Quaternion::Normalize)
        .def("GetNorm", &Quaternion::GetNorm)
        .def("RotationQuaternion", &Quaternion::RotationQuaternion,
            (bp::arg("ang"), bp::arg("v1"), bp::arg("v2"), bp::arg("v3")))
        .staticmethod("RotationQuaternion")
        .add_property("Q0", &getQ0, &setQ0)
        .add_property("Q1", &getQ1, &setQ1)
        .add_property("Q2", &getQ2, &setQ2)
        .add_property("Q3", &getQ3, &setQ3)
        .def(self * self)
        .def(self *= self)
        ;
}
