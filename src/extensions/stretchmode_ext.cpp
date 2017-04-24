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
* boost::python bindings to ObjCryst::StretchMode and its derivatives.
*
* Note that all indices are zero-based.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/list.hpp>
#include <boost/python/pure_virtual.hpp>

#include <set>

#include <ObjCryst/ObjCryst/Molecule.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


namespace {

typedef std::set<MolAtom*> MolAtomSet;

class StretchModeWrap : public StretchMode,
                        public wrapper<StretchMode>
{
    public:

    StretchModeWrap() : StretchMode() {}
    StretchModeWrap(const StretchModeWrap& sm) : StretchMode(sm) {}

    // Pure virtual

    void CalcDeriv(const bool derivllk=true) const
    {
        this->get_override("CalcDeriv")(derivllk);
    }

    void Stretch(const double change, const bool keepCenter)
    {
        this->get_override("Stretch")(change, keepCenter);
    }

    void RandomStretch(const double change, const bool keepCenter)
    {
        this->get_override("RandomStretch")(change);
    }

};

// These gives us a way to add an atom to a stretch mode
void _AddAtomSMBL(StretchModeBondLength& mode, MolAtom& a)
{
    mode.mvTranslatedAtomList.insert(&a);
}

void _AddAtomsSMBL(StretchModeBondLength& mode, bp::object& l)
{
    for(int i=0; i < len(l); ++i)
    {
        MolAtom* a = extract<MolAtom*>(l[i]);
        mode.mvTranslatedAtomList.insert(a);
    }
}

void _AddAtomsSetSMBL(StretchModeBondLength& mode, MolAtomSet& l)
{

    MolAtomSet::const_iterator p;
    for(p = l.begin(); p != l.end(); ++p)
    {
        mode.mvTranslatedAtomList.insert(*p);
    }
}

bp::list _GetAtomsSMBL(StretchModeBondLength& mode)
{
    bp::list l;

    MolAtomSet& v = mode.mvTranslatedAtomList;

    l = ptrcontainerToPyList< MolAtomSet >(v);

    return l;
}


// This one is for the angle modes
template <class T>
void _AddAtom(T& mode, MolAtom& a)
{
    mode.mvRotatedAtomList.insert(&a);
}

template <class T>
void _AddAtoms(T& mode, bp::object& l)
{
    for(int i=0; i < len(l); ++i)
    {
        MolAtom* a = extract<MolAtom*>(l[i]);
        mode.mvRotatedAtomList.insert(a);
    }
}

template <class T>
void _AddAtomsSet(T& mode, MolAtomSet& l)
{

    MolAtomSet::const_iterator p;
    for(p = l.begin(); p != l.end(); ++p)
    {
        mode.mvRotatedAtomList.insert(*p);
    }
}

template <class T>
bp::list _GetAtoms(T& mode)
{
    bp::list l;

    MolAtomSet& v = mode.mvRotatedAtomList;

    l = ptrcontainerToPyList< MolAtomSet >(v);

    return l;
}

// These are accessors for the atoms.

template <class T>
MolAtom* _GetAtom0(T& mode)
{
    return mode.mpAtom0;
}

template <class T>
MolAtom* _GetAtom1(T& mode)
{
    return mode.mpAtom1;
}

template <class T>
MolAtom* _GetAtom2(T& mode)
{
    return mode.mpAtom2;
}

} // namespace


void wrap_stretchmode()
{

    class_<StretchModeWrap, boost::noncopyable> ("StretchMode", no_init )
        .def("CalcDeriv", pure_virtual(&StretchMode::CalcDeriv),
            (bp::arg("derivllk")=true))
        .def("Stretch", pure_virtual(&StretchMode::Stretch),
            (bp::arg("amplitude"), bp::arg("keepCenter")=true))
        .def("RandomStretch", pure_virtual(&StretchMode::RandomStretch),
            bp::arg("amplitude"))
        ;

    class_<StretchModeBondLength, bases<StretchMode> >
        ("StretchModeBondLength",
        init<MolAtom&, MolAtom&, MolBond*>
        ((bp::arg("at0"), bp::arg("at1"), bp::arg("pBond")=0))
        [with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3,
                with_custodian_and_ward<1,4> > >()])
        .def("AddAtom", &_AddAtomSMBL,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtomsSMBL,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtomsSetSMBL,
            with_custodian_and_ward<1,2>())
        .def("GetAtoms", &_GetAtomsSMBL,
            with_custodian_and_ward_postcall<1,0>())
        .add_property("mpAtom0",
            make_function( &_GetAtom0<StretchModeBondLength>,
            return_internal_reference<>()))
        .add_property("mpAtom1",
            make_function( &_GetAtom1<StretchModeBondLength>,
            return_internal_reference<>()))
        ;

    class_<StretchModeBondAngle, bases<StretchMode> >
        ("StretchModeBondAngle",
        init<MolAtom&, MolAtom&, MolAtom&, MolBondAngle*>
        ((bp::arg("at0"), bp::arg("at1"), bp::arg("at2"),
          bp::arg("pBondAngle")=0))
        [with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3,
                with_custodian_and_ward<1,4,
                    with_custodian_and_ward<1,5> > > >()])
        .def("AddAtom", &_AddAtom<StretchModeBondAngle>,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtoms<StretchModeBondAngle>,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtomsSet<StretchModeBondAngle>,
            with_custodian_and_ward<1,2>())
        .def("GetAtoms", &_GetAtoms<StretchModeBondAngle>,
            with_custodian_and_ward_postcall<1,0>())
        .add_property("mpAtom0",
            make_function( &_GetAtom0<StretchModeBondAngle>,
            return_internal_reference<>()))
        .add_property("mpAtom1",
            make_function( &_GetAtom1<StretchModeBondAngle>,
            return_internal_reference<>()))
        .add_property("mpAtom2",
            make_function( &_GetAtom2<StretchModeBondAngle>,
            return_internal_reference<>()))
        ;

    class_<StretchModeTorsion, bases<StretchMode> >
        ("StretchModeTorsion",
        init<MolAtom&, MolAtom&, MolDihedralAngle*>
        ((bp::arg("at0"), bp::arg("at1"), bp::arg("pDihedralAngle")=0))
        [with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3,
                with_custodian_and_ward<1,4> > >()])
        .def("AddAtom", &_AddAtom<StretchModeTorsion>,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtoms<StretchModeTorsion>,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtomsSet<StretchModeTorsion>,
            with_custodian_and_ward<1,2>())
        .def("GetAtoms", &_GetAtoms<StretchModeTorsion>,
            with_custodian_and_ward_postcall<1,0>())
        .add_property("mpAtom1",
            make_function( &_GetAtom1<StretchModeTorsion>,
            return_internal_reference<>()))
        .add_property("mpAtom2",
            make_function( &_GetAtom2<StretchModeTorsion>,
            return_internal_reference<>()))
        ;

    class_<StretchModeTwist, bases<StretchMode> >
        ("StretchModeTwist",
        init<MolAtom&, MolAtom&>
        ((bp::arg("at0"), bp::arg("at1")))
        [with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3> >()])
        .def("AddAtom", &_AddAtom<StretchModeTwist>,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtoms<StretchModeTwist>,
            with_custodian_and_ward<1,2>())
        .def("AddAtoms", &_AddAtomsSet<StretchModeTwist>,
            with_custodian_and_ward<1,2>())
        .def("GetAtoms", &_GetAtoms<StretchModeTwist>,
            with_custodian_and_ward_postcall<1,0>())
        .add_property("mpAtom1",
            make_function( &_GetAtom1<StretchModeTwist>,
            return_internal_reference<>()))
        .add_property("mpAtom2",
            make_function( &_GetAtom2<StretchModeTwist>,
            return_internal_reference<>()))
        ;
}
