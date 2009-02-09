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
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <string>
#include <vector>
#include <iostream>

#include "converters.hpp"
#include <numpy/arrayobject.h>

#include "CrystVector/CrystVector.h"
#include "ObjCryst/Crystal.h"
#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/SpaceGroup.h"

using namespace boost::python;
using namespace ObjCryst;

namespace {
// for testing

CrystVector<float> getTestVector()
{ 
    /* Should produce
     * 0 1 2
     */
    CrystVector<float> tv(3);
    for(int i=0;i<3;i++)
    {
        tv(i) = i;
    }
    return tv;
}

CrystMatrix<float> getTestMatrix()
{ 
    /* Should produce
     * 0    1
     * 2    3
     * 4    5
     */
    CrystMatrix<float> tm(3,2);
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

typedef std::pair< ScatteringPower const*, ScatteringPower const* > sppair;

typedef std::map< sppair, float > mapsppairtofloat;

typedef std::map< sppair, Crystal::BumpMergePar > mapsppairtobmp; }

BOOST_PYTHON_MODULE(_registerconverters)
{

    import_array();
    to_python_converter< CrystVector<float>, CrystVector_REAL_to_ndarray >();
    to_python_converter< CrystMatrix<float>, CrystMatrix_REAL_to_ndarray >();
    // From boost sources
    std_pair_to_python_converter
        <ScatteringPower const *, ScatteringPower const * >();
    // Semi-converter for mapsppairtofloat
    class_<mapsppairtofloat>("mapsppairtofloat", no_init)
        .def(map_indexing_suite<mapsppairtofloat>());
    // Semi-converter for mapsppairtobmp
    class_<mapsppairtobmp>("mapsppairtobmp", no_init)
        .def(map_indexing_suite<mapsppairtobmp>());

    // some tests
    def("getTestVector", &getTestVector);
    def("getTestMatrix", &getTestMatrix);

}
