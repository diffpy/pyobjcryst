/*****************************************************************************
*
* pyobjcryst        Complex Modeling Initiative
*                   (c) 2015 Brookhaven Science Associates
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Kevin Knox
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::ScatteringData. This is a virtual class
* that can be derived from in python. These bindings are used by ObjCryst
* objects that inherit from ScatteringData (see, for example,
* diffractiondatasinglecrystal_ext.cpp).  ScatteringData derivatives can be
* created in python and will work in c++ functions that are also bound into
* python.
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <string>
#include <map>
#include <iostream>

#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/RefinableObj/IO.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers.hpp"
#include "python_file_stream.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {


class ScatteringDataWrap : public ScatteringData,
                 public wrapper<ScatteringData>
{

    public:

    void default_GenHKLFullSpace2(const REAL maxsithsl, const bool unique=false)
    { this->ScatteringData::GenHKLFullSpace2(maxsithsl, unique); }

    void GenHKLFullSpace2(const REAL maxsithsl, const bool unique=false)
    {
	if (override GenHKLFullSpace2 = this->get_override("GenHKLFullSpace2"))
	    GenHKLFullSpace2(maxsithsl, unique);
	default_GenHKLFullSpace2(maxsithsl, unique);
    }

    void default_SetCrystal(Crystal &crystal)
    { this->ScatteringData::SetCrystal(crystal); }

    void SetCrystal(Crystal &crystal)
    {
	if (override SetCrystal = this->get_override("SetCrystal"))
	    SetCrystal(crystal);
	default_SetCrystal(crystal);
    }

};

} //anonymous namespace

void wrap_scatteringdata()
{

    class_<ScatteringDataWrap, boost::noncopyable>("ScatteringData")
        /* Methods */
        .def("SetCrystal", &ScatteringData::SetCrystal,
            &ScatteringDataWrap::default_SetCrystal)
        .def("GenHKLFullSpace2", &ScatteringData::GenHKLFullSpace2,
            &ScatteringDataWrap::default_SetCrystal)
	;
}
