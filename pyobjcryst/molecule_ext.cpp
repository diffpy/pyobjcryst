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
* boost::python bindings to ObjCryst::Molecule.  
* 
* Changes from ObjCryst++
* - RemoveAtom returns None
* - RemoveBond returns None
* - RemoveBondAngle returns None
* - RemoveDihedralAngle returns None
* - FindBond returns the bond if found, None otherwise
* - FindBondAngle returns the bond angle if found, None otherwise
* - FindDihedralAngle returns the dihedral angle if found, None otherwise
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Molecule.h"
#include "ObjCryst/Crystal.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <vector>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overloaded for void return type
void _RemoveAtom(Molecule& m, MolAtom& ma)
{
    m.RemoveAtom(ma);
}

// Overloaded for void return type
void _RemoveBond(Molecule& m, const MolBond& mb)
{
    m.RemoveBond(mb);
}

// Overloaded for void return type
void _RemoveBondAngle(Molecule& m, MolBondAngle& mba)
{
    m.RemoveBondAngle(mba);
}

// Overloaded for void return type
void _RemoveDihedralAngle(Molecule& m, MolDihedralAngle& mda)
{
    m.RemoveDihedralAngle(mda);
}

PyObject* _FindBond(const Molecule& m, const MolAtom& ma1, const MolAtom& ma2)
{
    std::vector<MolBond*>::const_iterator mbi;
    mbi = m.FindBond(ma1, ma2);
    const std::vector<MolBond*> bondlist = m.GetBondList();
    PyObject *retval;
    if(bondlist.end() == mbi)
    {
        // return None
        retval = Py_None;
    }
    else
    {
        reference_existing_object::apply<MolBond*>::type converter;
        retval = converter(*mbi);
        
    }
    bp::incref(retval);
    return retval;
}

PyObject* _FindBondAngle(const Molecule& m, const MolAtom& ma1, const MolAtom&
    ma2, const MolAtom& ma3)
{
    std::vector<MolBondAngle*>::const_iterator mbai;
    mbai = m.FindBondAngle(ma1, ma2, ma3);
    const std::vector<MolBondAngle*> bondanglelist = m.GetBondAngleList();
    PyObject *retval;
    if(bondanglelist.end() == mbai)
    {
        // return None
        retval = Py_None;
    }
    else
    {
        reference_existing_object::apply<MolBondAngle*>::type converter;
        retval = converter(*mbai);
    }
    bp::incref(retval);
    return retval;
}

PyObject* _FindDihedralAngle(const Molecule& m, const MolAtom& ma1, 
    const MolAtom& ma2, const MolAtom& ma3, const MolAtom& ma4)
{
    std::vector<MolDihedralAngle*>::const_iterator mdai;
    mdai = m.FindDihedralAngle(ma1, ma2, ma3, ma4);
    const std::vector<MolDihedralAngle*> bondanglelist 
        = m.GetDihedralAngleList();
    PyObject *retval;
    if(bondanglelist.end() == mdai)
    {
        // return None
        retval = Py_None;
    }
    else
    {
        reference_existing_object::apply<MolDihedralAngle*>::type converter;
        retval = converter(*mdai);
    }
    bp::incref(retval);
    return retval;
}

} // namespace


BOOST_PYTHON_MODULE(_molecule)
{

    class_<Molecule, bases<Scatterer> > ("Molecule", 
        init<Crystal&, const string&> ((bp::arg("cryst"), bp::arg("name")="")))
        //[with_custodian_and_ward<2,1>()])
        // The crystal is not used, so we don't need to manage it.
        /* Constructors */
        .def(init<const Molecule&>((bp::arg("old"))))
        /* Methods */
        .def("AddAtom", &Molecule::AddAtom, 
            (bp::arg("x"), bp::arg("y"), bp::arg("z"), bp::arg("pPow"),
             bp::arg("name"), bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,5>())
        .def("RemoveAtom", &_RemoveAtom)
        .def("AddBond", &Molecule::AddBond,
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("length"),
             bp::arg("sigma"), bp::arg("delta"), bp::arg("bondOrder")=1,
             bp::arg("updateDisplay")=true))
        .def("RemoveBond", &_RemoveBond)
        .def("FindBond", &_FindBond, 
            with_custodian_and_ward_postcall<0,1>())
        .def("AddBondAngle", &Molecule::AddBondAngle, 
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("angle"), bp::arg("sigma"), bp::arg("delta"),
             bp::arg("updateDisplay")=true))
        .def("RemoveBondAngle", &_RemoveBondAngle)
        .def("FindBondAngle", &_FindBondAngle,
            with_custodian_and_ward_postcall<0,1>())
        .def("AddDihedralAngle", &Molecule::AddDihedralAngle, 
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("atom4"), bp::arg("angle"), bp::arg("sigma"),
             bp::arg("delta"), bp::arg("updateDisplay")=true))
        .def("RemoveDihedralAngle", &_RemoveDihedralAngle)
        .def("FindDihedralAngle", &_FindDihedralAngle,
            with_custodian_and_ward_postcall<0,1>())



        .def("GetAtom", 
            (MolAtom& (Molecule::*)(unsigned int)) &Molecule::GetAtom, 
            return_internal_reference<>())

        ;
}
