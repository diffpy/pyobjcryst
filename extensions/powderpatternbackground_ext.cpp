/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::PowderPattern.
*
* Changes from ObjCryst::MonteCarloObj
*
* Other Changes
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/slice.hpp>

#include <numpy/noprefix.h>
#include <numpy/arrayobject.h>
//#include "boost/python/extract.hpp"
//#include "boost/python/numeric.hpp"

#include <string>
#include <map>

#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {
  void _SetInterpPoints(PowderPatternBackground &b, PyObject *tth, PyObject *backgd)
  {
     cout<<"_SetInterpPoints:"<<tth<<", "<<backgd<<endl;
	 const unsigned long ndim=PyArray_NDIM((PyObject*)tth);
	 cout<<"dimensions="<<ndim<<endl;
	 const unsigned long nb=*(PyArray_DIMS((PyObject*)tth));
	 cout<<"nbPoints="<<nb<<endl;
	 CrystVector_REAL tth2(nb), backgd2(nb);
	 //:TODO: We assume the arrays are contiguous & double (float64) !
	 double *p=(double*) (PyArray_DATA(tth));
	 double *p2=(double*) (tth2.data());
	 for(unsigned long i=0;i<nb;i++) *p2++=*p++;
	 p=(double*) (PyArray_DATA(backgd));
	 p2=(double*) (backgd2.data());
	 for(unsigned long i=0;i<nb;i++) *p2++=*p++;
	 b.SetInterpPoints(tth2, backgd2);
  }

}   // namespace

void wrap_powderpatternbackground()
{
    class_<PowderPatternBackground, bases<PowderPatternComponent> >("PowderPatternBackground", init<>())
        //.def("SetParentPowderPattern", &PowderPatternBackground::SetParentPowderPattern)
        .def("GetPowderPatternCalc", &PowderPatternBackground::GetPowderPatternCalc,
                return_value_policy<copy_const_reference>())
        .def("ImportUserBackground", &PowderPatternBackground::ImportUserBackground,
                (bp::arg("filename")))
        .def("SetInterpPoints", _SetInterpPoints,
                (bp::arg("tth"), bp::arg("backgd")))
        .def("OptimizeBayesianBackground", &PowderPatternBackground::OptimizeBayesianBackground)
        .def("FixParametersBeyondMaxresolution",
                &PowderPatternBackground::FixParametersBeyondMaxresolution, (bp::arg("obj")))
        ;
}
