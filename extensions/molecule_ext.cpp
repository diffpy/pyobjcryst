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
* boost::python bindings to ObjCryst::Molecule.
*
* Note that all indices are zero-based.
*
* Changes from ObjCryst::Molecule
* - The public data are not wrapped.
* - Added __getitem__ access for MolAtoms.
* - AddAtom returns the added MolAtom
* - AddBond returns the added MolBond
* - AddBondAngle returns the added MolBondAngle
* - AddDihedralAngle returns the added MolDihedralAngle
* - RemoveAtom returns None, has an indexed version
* - RemoveBond returns None, has an indexed version
* - RemoveBondAngle returns None, has an indexed version
* - RemoveDihedralAngle returns None, has an indexed version
* - RemoveRigidGroup returns None
* - Added GetNbAtoms
* - Added GetNbBonds
* - Added GetNbBondAngles
* - Added GetNbDihedralAngles
* - Added GetNbRigidGroups
* - Added GetBond
* - Added GetBondAngle
* - Added GetDihedralAngle
* - Added GetRigidGroup
* - FindBond returns the bond if found, None otherwise
* - FindBondAngle returns the bond angle if found, None otherwise
* - FindDihedralAngle returns the dihedral angle if found, None otherwise
* - FindAtom is identical to GetAtom.
* - FlipAtomGroup is not wrapped.
* - FlipGroup, RotorGroup and StretchModeGroup are not wrapped.
* - StretchMode getters are not wrapped.
* - Quaternion ordinates Q0, Q1, Q2 and Q3 wrapped as properties.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/slice.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Molecule.h>
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


namespace {


// Overloaded to return added object and manage the lifetime of the object.
MolAtom& _AddAtom(Molecule& m, const double x, const double y, const double z,
        const ScatteringPower* pow, const std::string& name,
        const bool updateDisplay=true)
{
    m.AddAtom(x, y, z, pow, name, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    std::vector<MolAtom*>& v = m.GetAtomList();
    return *v.back();
}

MolBond& _AddBond(Molecule& m, MolAtom& atom1, MolAtom& atom2, const double
        length, const double sigma, const double delta, const double bondOrder =
        1., const bool updateDisplay = true)
{
    m.AddBond(atom1, atom2, length, sigma, delta, bondOrder, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    std::vector<MolBond*>& v = m.GetBondList();
    return *v.back();
}

MolBondAngle& _AddBondAngle(Molecule& m, MolAtom& atom1, MolAtom& atom2,
        MolAtom& atom3, const double angle, const double sigma, const double
        delta, const bool updateDisplay = true)
{
    m.AddBondAngle(atom1, atom2, atom3, angle, sigma, delta, updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    std::vector<MolBondAngle*>& v = m.GetBondAngleList();
    return *v.back();
}

MolDihedralAngle& _AddDihedralAngle(Molecule& m, MolAtom& atom1, MolAtom&
        atom2, MolAtom& atom3, MolAtom& atom4, const double angle, const double
        sigma, const double delta, const bool updateDisplay = true)
{
    m.AddDihedralAngle(atom1, atom2, atom3, atom4, angle, sigma, delta,
            updateDisplay);
    m.SetDeleteSubObjInDestructor(false);
    std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();
    return *v.back();
}

// New Functions
size_t _GetNbAtoms(Molecule& m)
{
    std::vector<MolAtom*>& v = m.GetAtomList();
    return v.size();
}

size_t _GetNbBonds(Molecule& m)
{
    std::vector<MolBond*>& v = m.GetBondList();
    return v.size();
}

size_t _GetNbBondAngles(Molecule& m)
{
    std::vector<MolBondAngle*>& v = m.GetBondAngleList();
    return v.size();
}

size_t _GetNbDihedralAngles(Molecule& m)
{
    std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();
    return v.size();
}

size_t _GetNbRigidGroups(Molecule& m)
{
    std::vector<RigidGroup*>& v = m.GetRigidGroupList();
    return v.size();
}

// Overloaded for safety
MolAtom& _GetAtomIdx(Molecule& m, int idx)
{
    std::vector<MolAtom*>& v = m.GetAtomList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= (int) v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
}

MolBond& _GetBondIdx(Molecule& m, int idx)
{
    std::vector<MolBond*>& v = m.GetBondList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= (int) v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
}

MolBondAngle& _GetBondAngleIdx(Molecule& m, int idx)
{
    std::vector<MolBondAngle*>& v = m.GetBondAngleList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= (int) v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
}

MolDihedralAngle& _GetDihedralAngleIdx(Molecule& m, int idx)
{
    std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= (int) v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
}

// Overloaded for void return type and index access.
void _RemoveAtom(Molecule& m, MolAtom& ma)
{
    m.RemoveAtom(ma, false);
}

void _RemoveAtomIdx(Molecule& m, int idx)
{
    m.RemoveAtom(_GetAtomIdx(m, idx), false);
}

void _RemoveBond(Molecule& m, const MolBond& mb)
{
    m.RemoveBond(mb, false);
}

void _RemoveBondIdx(Molecule& m, int idx)
{
    m.RemoveBond(_GetBondIdx(m, idx), false);
}

void _RemoveBondAngle(Molecule& m, MolBondAngle& mba)
{
    m.RemoveBondAngle(mba, false);
}

void _RemoveBondAngleIdx(Molecule& m, int idx)
{
    m.RemoveBondAngle(_GetBondAngleIdx(m, idx), false);
}

void _RemoveDihedralAngle(Molecule& m, MolDihedralAngle& mda)
{
    m.RemoveDihedralAngle(mda, false);
}

void _RemoveDihedralAngleIdx(Molecule& m, int idx)
{
    m.RemoveDihedralAngle(_GetDihedralAngleIdx(m, idx), false);
}

// Overloaded to return new rigid group object
RigidGroup& _AddRigidGroup(Molecule& m, const RigidGroup& r, const bool ud=true)
{
    m.AddRigidGroup(r, ud);
    // Get the new rigid group and return it
    std::vector<RigidGroup*>& v = m.GetRigidGroupList();
    return *v.back();
}

// Overloaded to accept python iterable and to return the rigid group object
RigidGroup& _AddRigidGroupIterable(Molecule& m, bp::object& l, const bool ud=true)
{
    // convert l to a rigid group
    RigidGroup* r = new RigidGroup();

    for(int i=0; i < len(l); ++i)
    {
        MolAtom* a = extract<MolAtom*>(l[i]);
        r->insert(a);
    }

    // Add this rigid group and delete it, since AddRigidGroup makes a copy.
    m.AddRigidGroup(*r, ud);
    delete r;

    // Get the new rigid group and return it
    std::vector<RigidGroup*>& v = m.GetRigidGroupList();
    return *v.back();
}

// Overloaded for void return type
void _RemoveRigidGroup(Molecule& m, RigidGroup& mda, const bool ud=true)
{
    m.RemoveRigidGroup(mda, ud, false);
}

PyObject* _FindBond(const Molecule& m, const MolAtom& ma1, const MolAtom& ma2)
{
    std::vector<MolBond*>::const_iterator mbi;
    mbi = m.FindBond(ma1, ma2);
    const std::vector<MolBond*>& bondlist = m.GetBondList();
    PyObject* retval;
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
    const std::vector<MolBondAngle*>& bondanglelist = m.GetBondAngleList();
    PyObject* retval;
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
    const std::vector<MolDihedralAngle*>& dihedralanglelist
        = m.GetDihedralAngleList();
    PyObject* retval;
    if(dihedralanglelist.end() == mdai)
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

// This could usually be done with an indexing suite, but it doesn't work with
// vectors of pointers.
bp::list _GetAtomList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolAtom*>& v = m.GetAtomList();

    l = ptrcontainerToPyList< const std::vector<MolAtom*> >(v);

    return l;
}

bp::list _GetBondList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolBond*>& v = m.GetBondList();

    l = ptrcontainerToPyList< const std::vector<MolBond*> >(v);

    return l;
}

bp::list _GetBondAngleList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolBondAngle*>& v = m.GetBondAngleList();

    l = ptrcontainerToPyList< const std::vector<MolBondAngle*> >(v);

    return l;
}

bp::list _GetDihedralAngleList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();

    l = ptrcontainerToPyList< const std::vector<MolDihedralAngle*> >(v);

    return l;
}

bp::list _GetStretchModeBondLengthList(const Molecule& m)
{
    bp::list l;

    const std::list<StretchModeBondLength>& v =
        m.GetStretchModeBondLengthList();

    l = containerToPyList< const std::list<StretchModeBondLength> >(v);

    return l;
}

bp::list _GetStretchModeBondAngleList(const Molecule& m)
{
    bp::list l;

    const std::list<StretchModeBondAngle>& v = m.GetStretchModeBondAngleList();

    l = containerToPyList< const std::list<StretchModeBondAngle> >(v);

    return l;
}

bp::list _GetStretchModeTorsionList(const Molecule& m)
{
    bp::list l;

    const std::list<StretchModeTorsion>& v = m.GetStretchModeTorsionList();

    l = containerToPyList< const std::list<StretchModeTorsion> >(v);

    return l;
}

bp::list _GetRigidGroupList(const Molecule& m)
{
    bp::list l;

    const std::vector<RigidGroup*>& v = m.GetRigidGroupList();

    l = ptrcontainerToPyList< const std::vector<RigidGroup*> >(v);

    return l;
}

// Get atoms by slice
bp::object getAtomSlice(Molecule& m, bp::slice& s)
{
    bp::list l = _GetAtomList(m);
    return l[s];
}

// Overloaded to accept a python iterable instead of a std::set. Again, could be
// done with converters, but there are issues with pointers. Perhaps another
// day...
void _RotateAtomGroup(Molecule& m, const MolAtom& at1, const MolAtom& at2,
    const bp::object& atoms, const double angle, const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyIterableToSet<MolAtom*>(atoms);
    m.RotateAtomGroup(at1, at2, catoms, angle, keepCenter);
}

void _RotateAtomGroupVec(Molecule& m, const MolAtom& at1, const double vx,
    const double vy, const double vz, const bp::object& atoms, const double angle,
    const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyIterableToSet<MolAtom*>(atoms);
    m.RotateAtomGroup(at1, vx, vy, vz, catoms, angle, keepCenter);
}

// A new method for three-tuples
void _RotateAtomGroup2Vec(Molecule& m, bp::object& v1, bp::object& v2,
    const bp::object& atoms, const double angle,
    const bool keepCenter=true)
{

    double x, y, z;
    x = extract<double>(v1[0]);
    y = extract<double>(v1[1]);
    z = extract<double>(v1[2]);
    MolAtom& a = _AddAtom(m, x, y, z, 0, "_rag2vectemp", false);
    x = extract<double>(v2[0]);
    y = extract<double>(v2[1]);
    z = extract<double>(v2[2]);
    _RotateAtomGroupVec(m, a, x, y, z, atoms, angle, keepCenter);
    m.RemoveAtom(a, true);
    return;
}

void _TranslateAtomGroup(Molecule& m, const bp::object& atoms, const double dx,
    const double dy, const double dz, const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyIterableToSet<MolAtom*>(atoms);
    m.TranslateAtomGroup(catoms, dx, dy, dz, keepCenter);
}

bp::dict _GetConnectivityTable(Molecule& m)
{

    const std::map<MolAtom*, std::set<MolAtom*> >& ct
        = m.GetConnectivityTable();

    std::map<MolAtom*, std::set<MolAtom*> >::const_iterator miter;

    bp::dict d;

    for(miter = ct.begin(); miter != ct.end(); ++miter)
    {
        bp::object key(bp::ptr(miter->first));

        d[key] = ptrcontainerToPyList< const std::set<MolAtom*>
            >(miter->second);
    }

    return d;
}

bp::list _AsZMatrix(const Molecule& m, const bool keeporder)
{
    bp::list l;

    const std::vector<MolZAtom>& v = m.AsZMatrix(keeporder);

    l = containerToPyList< const std::vector<MolZAtom> >(v);

    return l;
}

std::string quatparname(const Molecule& m, int idx)
{
    using namespace std;
    static bool didseparator = false;
    static bool prefixmolname = false;
    static string separator;
    if (!didseparator)
    {
        map<string,int> qnames;
        for (long i = 0; i < m.GetNbPar(); ++i)
        {
            const string& pname = m.GetPar(i).GetName();
            size_t n = pname.size();
            if (n < 2)  continue;
            if (pname[n - 2] != 'Q')  continue;
            if (pname.find_last_of("0123", n - 1) == string::npos)  continue;
            qnames[pname.substr(0, n - 2)] += 1;
        }
        map<string,int>::iterator qni;
        const string& mname = m.GetName();
        for (qni = qnames.begin(); qni != qnames.end(); ++qni)
        {
            if (qni->second == 4)
            {
                const string& qnm = qni->first;
                prefixmolname = (qnm.size() >= mname.size() &&
                        qnm.substr(0, mname.size()) == mname);
                size_t p0 = prefixmolname ? mname.size() : 0;
                separator = qnm.substr(p0);
                didseparator = true;
            }
        }
    }
    ostringstream rv;
    rv << (prefixmolname ? m.GetName() : "") << separator << 'Q' << idx;
    return rv.str();
}


// Setters and getters for position
void _setQ0(Molecule& m, double val)
{
    m.GetPar(quatparname(m, 0)).SetValue(val);
}

double _getQ0(Molecule& m)
{
    return m.GetPar(quatparname(m, 0)).GetValue();
}

void _setQ1(Molecule& m, double val)
{
    m.GetPar(quatparname(m, 1)).SetValue(val);
}

double _getQ1(Molecule& m)
{
    return m.GetPar(quatparname(m, 1)).GetValue();
}

void _setQ2(Molecule& m, double val)
{
    m.GetPar(quatparname(m, 2)).SetValue(val);
}

double _getQ2(Molecule& m)
{
    return m.GetPar(quatparname(m, 2)).GetValue();
}

void _setQ3(Molecule& m, double val)
{
    m.GetPar(quatparname(m, 3)).SetValue(val);
}

double _getQ3(Molecule& m)
{
    return m.GetPar(quatparname(m, 3)).GetValue();
}


} // namespace


void wrap_molecule()
{

    class_<Molecule, bases<Scatterer> > ("Molecule",
        init<const Molecule&>(bp::arg("oldMolecule")))
        /* Constructors */
        .def(init<Crystal&, const std::string&>(
            (bp::arg("cryst"), bp::arg("name")="")))
        /* Methods */
        .def("GetFormula", &Molecule::GetFormula)
        .def("AddAtom", &_AddAtom,
            (bp::arg("x"), bp::arg("y"), bp::arg("z"), bp::arg("pPow"),
             bp::arg("name"), bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,5, return_internal_reference<> >())
        .def("RemoveAtom", &_RemoveAtom)
        .def("RemoveAtom", &_RemoveAtomIdx)
        .def("AddBond", &_AddBond,
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("length"),
             bp::arg("sigma"), bp::arg("delta"), bp::arg("bondOrder")=1,
             bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3,
            return_internal_reference<> > >())
        .def("RemoveBond", &_RemoveBond)
        .def("RemoveBond", &_RemoveBondIdx)
        .def("GetBond", &_GetBondIdx, return_internal_reference<>())
        .def("FindBond", &_FindBond,
            with_custodian_and_ward_postcall<1,0>())
        .def("AddBondAngle", &_AddBondAngle,
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("angle"), bp::arg("sigma"), bp::arg("delta"),
             bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3,
            with_custodian_and_ward<1,4,
            return_internal_reference<> > > >())
        .def("RemoveBondAngle", &_RemoveBondAngle)
        .def("RemoveBondAngle", &_RemoveBondAngleIdx)
        .def("GetBondAngle", &_GetBondAngleIdx, return_internal_reference<>())
        .def("FindBondAngle", &_FindBondAngle,
            with_custodian_and_ward_postcall<1,0>())
        .def("AddDihedralAngle", &_AddDihedralAngle,
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("atom4"), bp::arg("angle"), bp::arg("sigma"),
             bp::arg("delta"), bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,2,
            with_custodian_and_ward<1,3,
            with_custodian_and_ward<1,4,
            with_custodian_and_ward<1,5,
            return_internal_reference<> > > > >())
        .def("RemoveDihedralAngle", &_RemoveDihedralAngle)
        .def("RemoveDihedralAngle", &_RemoveDihedralAngleIdx)
        .def("GetDihedralAngle", &_GetDihedralAngleIdx,
            return_internal_reference<>())
        .def("FindDihedralAngle", &_FindDihedralAngle,
            with_custodian_and_ward_postcall<1,0>())
        .def("AddRigidGroup", &_AddRigidGroup,
            (bp::arg("group"), bp::arg("updateDisplay") = true),
            return_internal_reference<>())
        .def("AddRigidGroup", &_AddRigidGroupIterable,
            (bp::arg("group"), bp::arg("updateDisplay") = true),
            return_internal_reference<>())
        .def("RemoveRigidGroup", &_RemoveRigidGroup,
                (bp::arg("group"), bp::arg("updateDisplay") = true))
        .def("GetAtom", &_GetAtomIdx, return_internal_reference<>())
        .def("GetAtom",
            (MolAtom& (Molecule::*)(const string&)) &Molecule::GetAtom,
            return_internal_reference<>())
        .def("FindAtom",
            (MolAtom& (Molecule::*)(const string&)) &Molecule::GetAtom,
            return_internal_reference<>())
        .def("OptimizeConformation", &Molecule::OptimizeConformation,
            (bp::arg("nbTrial")=10000, bp::arg("stopCost")=0))
        .def("OptimizeConformationSteepestDescent",
            &Molecule::OptimizeConformationSteepestDescent,
            (bp::arg("maxStep")=0.1, bp::arg("nbSteps")=1))
        .def("GetNbAtoms", &_GetNbAtoms)
        .def("GetNbBonds", &_GetNbBonds)
        .def("GetNbBondAngles", &_GetNbBondAngles)
        .def("GetNbDihedralAngles", &_GetNbDihedralAngles)
        .def("GetNbRigidGroups", &_GetNbRigidGroups)
        .def("GetAtomList", &_GetAtomList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondList", &_GetBondList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondAngleList", &_GetBondAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetDihedralAngleList", &_GetDihedralAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetRigidGroupList", &_GetRigidGroupList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeBondLengthList", &_GetStretchModeBondLengthList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeBondAngleList", &_GetStretchModeBondAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeTorsionList", &_GetStretchModeTorsionList,
            with_custodian_and_ward_postcall<1,0>())
        .def("RotateAtomGroup", &_RotateAtomGroup,
            (bp::arg("at1"), bp::arg("at2"), bp::arg("atoms"), bp::arg("angle"),
             bp::arg("keepCenter")=true
             )
            )
        .def("RotateAtomGroup", &_RotateAtomGroupVec,
            (bp::arg("at1"), bp::arg("vx"), bp::arg("vy"), bp::arg("vz"),
             bp::arg("atoms"), bp::arg("angle"), bp::arg("keepCenter")=true
             )
            )
        .def("RotateAtomGroup", &_RotateAtomGroup2Vec,
            (bp::arg("v1"), bp::arg("v2"), bp::arg("atoms"), bp::arg("angle"),
             bp::arg("keepCenter")=true
             )
            )
        .def("TranslateAtomGroup", &_TranslateAtomGroup,
            (bp::arg("atoms"), bp::arg("dx"), bp::arg("dy"), bp::arg("dz"),
             bp::arg("keepCenter")=true
             )
            )
        .def("GetConnectivityTable", &_GetConnectivityTable,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondListClock", (RefinableObjClock& (Molecule::*)())
            &Molecule::GetBondListClock,
            return_internal_reference<>())
        .def("GetAtomPositionClock", (RefinableObjClock& (Molecule::*)())
            &Molecule::GetAtomPositionClock,
            return_internal_reference<>())
        .def("GetRigidGroupClock", (RefinableObjClock& (Molecule::*)())
            &Molecule::GetRigidGroupClock,
            return_internal_reference<>())
        .def("RigidifyWithDihedralAngles",
            &Molecule::RigidifyWithDihedralAngles)
        .def("BondLengthRandomChange", &Molecule::BondLengthRandomChange,
            (bp::arg("mode"), bp::arg("amplitude"),
             bp::arg("respectRestraint")=true)
            )
        .def("BondAngleRandomChange", &Molecule::BondAngleRandomChange,
            (bp::arg("mode"), bp::arg("amplitude"),
             bp::arg("respectRestraint")=true)
            )
        .def("DihedralAngleRandomChange", &Molecule::DihedralAngleRandomChange,
            (bp::arg("mode"), bp::arg("amplitude"),
             bp::arg("respectRestraint")=true)
            )
        .def("GetCenterAtom", &Molecule::GetCenterAtom,
            return_internal_reference<>())
        // Memory management shouldn't be necessary here, but there is the
        // possibility that a MolAtom that was created in another Molecule is
        // passed to this one. This could lead to memory corruption if the
        // original Molecule was to be deleted before this one, hence the
        // with_custodian_and_ward.
        .def("SetCenterAtom", &Molecule::SetCenterAtom,
            with_custodian_and_ward<1,2>())
        .def("AsZMatrix", &_AsZMatrix,
            with_custodian_and_ward_postcall<1,0>())
        .def("BuildRingList", &Molecule::BuildRingList)
        .def("BuildConnectivityTable", &Molecule::BuildConnectivityTable)
        .def("BuildRotorGroup", &Molecule::BuildRotorGroup)
        .def("TuneGlobalOptimRotationAmplitude",
            &Molecule::TuneGlobalOptimRotationAmplitude)
        .def("BuildFlipGroup", &Molecule::BuildFlipGroup)
        .def("BuildStretchModeBondLength",
            &Molecule::BuildStretchModeBondLength)
        .def("BuildStretchModeBondAngle",
            &Molecule::BuildStretchModeBondAngle)
        .def("BuildStretchModeTorsion",
            &Molecule::BuildStretchModeTorsion)
        .def("BuildStretchModeTwist",
            &Molecule::BuildStretchModeTwist)
        .def("BuildStretchModeGroups",
            &Molecule::BuildStretchModeGroups)
        .def("UpdateScattCompList", &Molecule::UpdateScattCompList)
        .def("InitOptions", &Molecule::InitOptions)
        .def("__getitem__", &getAtomSlice,
                with_custodian_and_ward_postcall<1,0>())
        .def("__getitem__", &_GetAtomIdx,
                return_internal_reference<>())
        .def("__len__", &_GetNbAtoms)
        // Properties for molecule position
        .add_property("Q0", &_getQ0, &_setQ0)
        .add_property("Q1", &_getQ1, &_setQ1)
        .add_property("Q2", &_getQ2, &_setQ2)
        .add_property("Q3", &_getQ3, &_setQ3)
        ;

    // Wrap some functions
    def("GetBondLength", &GetBondLength);
    def("GetBondAngle", &GetBondAngle);
    def("GetDihedralAngle", &GetDihedralAngle);

}
