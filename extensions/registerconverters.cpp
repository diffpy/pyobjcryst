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
* boost::python bindings for various conversions used in pyobjcryst.
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/to_python_converter.hpp>

#include <vector>
#include <iostream>

#include <numpy/noprefix.h>
#include <numpy/arrayobject.h>

#include <ObjCryst/CrystVector/CrystVector.h>
#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/ObjCryst/SpaceGroup.h>
#include <ObjCryst/ObjCryst/Molecule.h>

#include "python_file_stream.hpp"


using namespace boost::python;
using namespace ObjCryst;


namespace {

namespace bp = boost::python;

typedef std::pair< ScatteringPower const*, ScatteringPower const* > sppair;

typedef std::map< sppair, double > mapsppairtodouble;

typedef std::map< sppair, Crystal::BumpMergePar > mapsppairtobmp;

typedef std::set<MolAtom*> MolAtomSet;

typedef std::vector<MolAtom*> MolAtomVec;


// Make an array out of a data pointer and a dimension vector
PyObject* makeNdArray(double * data, std::vector<npy_intp>& dims)
{
    PyObject* pyarray = PyArray_SimpleNewFromData
                (dims.size(), &dims[0], PyArray_DOUBLE, (void *) data);
    PyObject* pyarraycopy = PyArray_Copy( (PyArrayObject*) pyarray );
    return bp::incref(pyarraycopy);
}

// CrystVector to ndarray
struct CrystVector_REAL_to_ndarray
{

    static PyObject* convert(CrystVector<double> const &cv)
    {
        std::vector<npy_intp> dims(1);
        dims[0] = cv.numElements();
        return makeNdArray((double *) cv.data(), dims);
    }

    static PyTypeObject const* get_pytype()
    {
        return &PyDoubleArrType_Type;
    }

};

// CrystMatrix to ndarray
struct CrystMatrix_REAL_to_ndarray
{

    static PyObject* convert(CrystMatrix<double> const &cm)
    {
        std::vector<npy_intp> dims(2);
        dims[0] = cm.rows();
        dims[1] = cm.cols();
        return makeNdArray((double *) cm.data(), dims);
    }

    static PyTypeObject const* get_pytype()
    {
        return &PyDoubleArrType_Type;
    }

};

// std::pair to tuple
template <typename T1, typename T2>
struct std_pair_to_tuple
{

    static PyObject* convert(std::pair<T1, T2> const& p)
    {
        bp::object tpl = bp::make_tuple(p.first, p.second);
        PyObject* rv = tpl.ptr();
        return bp::incref(rv);
    }

    static PyTypeObject const* get_pytype()
    {
        return &PyTuple_Type;
    }

};

// Helper for convenience.
template <typename T1, typename T2>
struct std_pair_to_python_converter
{

std_pair_to_python_converter()
{
    bp::to_python_converter<
        std::pair<T1, T2>,
        std_pair_to_tuple<T1, T2>
    >();
}

};

/* For MolAtomSet (std::set<MolAtom*>) */

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

/* Exception translation */

PyObject* pyobjcryst_ObjCrystException =
    PyErr_NewException((char*)"pyobjcryst.ObjCrystException", 0, 0);


void translateException(const ObjCrystException& e)
{
    PyErr_SetString(pyobjcryst_ObjCrystException, e.message.c_str());
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

// From cctbx. Used to convert python file-type objects to c++ streams.
// See python_file_buffer.hpp for copyright.
namespace boost_adaptbx { namespace file_conversion {

  std::size_t python_file_buffer::buffer_size = 1024;

  // Boost.Python conversion dark magic
  struct python_file_to_stream_buffer
  {
    static void register_conversion() {
      using namespace boost::python;
      converter::registry::push_back(
        &convertible,
        &construct,
        type_id<python_file_buffer>());
    }

    static void *convertible(PyObject *obj_ptr) {
      using namespace boost::python;
      if (!(   PyObject_HasAttrString(obj_ptr, "read")
            && PyObject_HasAttrString(obj_ptr, "readline")
            && PyObject_HasAttrString(obj_ptr, "readlines"))
          &&
          !(   PyObject_HasAttrString(obj_ptr, "write")
            && PyObject_HasAttrString(obj_ptr, "writelines"))) return 0;
      return obj_ptr;
    }

    static void construct(
      PyObject *obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data *data)
    {
      using namespace boost::python;
      typedef converter::rvalue_from_python_storage<python_file_buffer> rvalue_t;
      void *storage = ((rvalue_t *) data)->storage.bytes;
      object python_file((handle<>(borrowed(obj_ptr))));
      new (storage) python_file_buffer(python_file);
      data->convertible = storage;
    }
  };

  struct python_file_buffer_wrapper
  {
    typedef python_file_buffer wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt, boost::noncopyable>("buffer", no_init)
        .def_readwrite("size", wt::buffer_size,
                       "The size of the buffer sitting "
                       "between a Python file object and a C++ stream.")
      ;
    }
  };

}} // boost_adaptbx::file_conversions


void wrap_registerconverters()
{

    /* Exceptions */
    register_exception_translator<ObjCrystException>(translateException);
    // We want silent exceptions
    ObjCrystException::verbose = false;


    // Put ObjCrystException in module namespace
    scope().attr("ObjCrystException") =
        object(handle<>(pyobjcryst_ObjCrystException));

    /* Data type converters */
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

    // Python file stuff
    namespace bafc = boost_adaptbx::file_conversion;
    bafc::python_file_to_stream_buffer::register_conversion();
    bafc::python_file_to_stream_buffer::register_conversion();
    bafc::python_file_buffer_wrapper::wrap();


    // some tests
    def("getTestVector", &getTestVector);
    def("getTestMatrix", &getTestMatrix);

}
