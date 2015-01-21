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
* boost::python bindings to ObjCryst::DiffractionDataSingleCrystal.
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

#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/RefinableObj/IO.h>
#include <ObjCryst/CrystVector/CrystVector.h>

#include "helpers.hpp"
#include "python_file_stream.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

void wrap_diffractiondatasinglecrystal()
{

    class_<DiffractionDataSingleCrystal, bases<ScatteringData> > 
	("DiffractionDataSingleCrystal", init<const DiffractionDataSingleCrystal&>())
        /* Methods */
        .def("SetWavelength", (void (DiffractionDataSingleCrystal::*)(const double))
		    &DiffractionDataSingleCrystal::SetWavelength)
	;
}
