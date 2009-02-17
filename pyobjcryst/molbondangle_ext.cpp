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
* boost::python bindings to ObjCryst::MolBondAngle.  
* 
* Changes from ObjCryst++
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Molecule.h"
#include "RefinableObj/RefinableObj.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <string>
#include <sstream>
#include <map>
#include <set>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

} // namespace


BOOST_PYTHON_MODULE(_molbondangle)
{

    class_<MolBondAngle, bases<Restraint> > ("MolBondAngle", no_init)
        .def("GetName", &MolBondAngle::GetName)
        ;
}
