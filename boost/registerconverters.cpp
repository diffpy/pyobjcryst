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
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "converters.hpp"
#include <numpy/arrayobject.h>

#include "CrystVector/CrystVector.h"
#include "ObjCryst/Crystal.h"
#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/SpaceGroup.h"
#include "ObjCryst/Molecule.h"

using namespace boost::python;
using namespace ObjCryst;

namespace {
// for testing


typedef std::pair< ScatteringPower const*, ScatteringPower const* > sppair;

typedef std::map< sppair, double > mapsppairtodouble;

typedef std::map< sppair, Crystal::BumpMergePar > mapsppairtobmp; 

typedef std::set<MolAtom*> MolAtomSet;

typedef std::vector<MolAtom*> MolAtomVec;

void _addMAS(MolAtomSet& mas, MolAtom* a) 
{ 
    mas.insert(a); 
}

void _updateMAS(MolAtomSet& mas, const MolAtomSet& other)
{
    mas.insert(other.begin(), other.end());
}

bool _containsMAS(const MolAtomSet& mas, MolAtom* a)
{
    return mas.find(a) != mas.end();
}

MolAtom* _getItemMAS(const MolAtomSet& mas, size_t i)
{
    // Look for size violation
    if (i >= mas.size()) 
    {        
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }

    MolAtomSet::const_iterator p = mas.begin();
    while (i > 0) { p++; i--; }
    return *p;
}

void _discardMAS(MolAtomSet& mas,  MolAtom* a)
{
    if( _containsMAS(mas, a) )
    {
        mas.erase(a);
    }
}

void _removeMAS(MolAtomSet& mas,  MolAtom* a)
{
    if( _containsMAS(mas, a) )
    {
        mas.erase(a);
    }
    else
    {
        PyErr_SetString(PyExc_KeyError, "KeyError");
        throw_error_already_set();
    }
}

/* For MolAtomVec */
void _appendMAV(MolAtomVec& mav, MolAtom* a) 
{ 
    mav.push_back(a); 
}

void _extendMAV(MolAtomVec& mav, const MolAtomVec& other)
{
    for(MolAtomVec::iterator p; p < other.end(); ++p)
    {
        mav.push_back(*p);
    }
}

bool _containsMAV(const MolAtomVec& mav, MolAtom* a)
{
    return std::find(mav.begin(), mav.end(), a) != mav.end();
}

MolAtom* _getItemMAV(const MolAtomVec& mav, size_t i)
{
    return mav[i];
}

void _setItemMAV(MolAtomVec& mav, size_t i, MolAtom* a)
{
    mav[i] = a;
}

void _deleteMAV(MolAtomVec& mav, size_t i)
{
    mav.erase(mav.begin()+i);
}


//
// For testing
//

CrystVector<double> getTestVector()
{ 
    /* Should produce
     * 0 1 2
     */
    CrystVector<double> tv(3);
    for(int i=0;i<3;i++)
    {
        tv(i) = i;
    }
    return tv;
}

CrystMatrix<double> getTestMatrix()
{ 
    /* Should produce
     * 0    1
     * 2    3
     * 4    5
     */
    CrystMatrix<double> tm(3,2);
    int counter = 0;
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<2;j++)
        {
            tm(i,j) = counter;
            counter++;
        }
    }
    return tm;
}

} // namespace


void wrap_registerconverters()
{

    import_array();
    to_python_converter< CrystVector<double>, CrystVector_REAL_to_ndarray >();
    to_python_converter< CrystMatrix<double>, CrystMatrix_REAL_to_ndarray >();
    // From boost sources
    std_pair_to_python_converter
        <ScatteringPower const *, ScatteringPower const * >();
    // Semi-converter for mapsppairtodouble
    class_<mapsppairtodouble>("mapsppairtodouble", no_init)
        .def(map_indexing_suite<mapsppairtodouble>());
    // Semi-converter for mapsppairtobmp
    class_<mapsppairtobmp>("mapsppairtobmp", no_init)
        .def(map_indexing_suite<mapsppairtobmp>());

    class_< MolAtomSet >("MolAtomSet", no_init)
        .def("add", &_addMAS, with_custodian_and_ward<1,2>())
        .def("clear", &std::set<MolAtom*>::clear)
        .def("discard", &_discardMAS)
        .def("remove", &_removeMAS)
        .def("update", &_updateMAS, with_custodian_and_ward<1,2>())
        .def("__contains__", &_containsMAS)
        .def("__getitem__", &_getItemMAS, return_internal_reference<>())
        .def("__len__", &MolAtomSet::size)
        ;

    class_< MolAtomVec >("MolAtomVec", no_init)
        .def("append", &_appendMAV, with_custodian_and_ward<1,2>())
        .def("extend", &_extendMAV, with_custodian_and_ward<1,2>())
        .def("contains", &_containsMAV)
        .def("delete", &_deleteMAV)
        .def("__getitem__", &_getItemMAV, return_internal_reference<>())
        .def("__setitem__", &_setItemMAV, with_custodian_and_ward<1,2>())
        .def("__len__", &MolAtomVec::size)
        ;

    // some tests
    def("getTestVector", &getTestVector);
    def("getTestMatrix", &getTestMatrix);

}
