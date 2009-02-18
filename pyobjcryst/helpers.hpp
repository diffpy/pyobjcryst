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
* This is home to converters that are explicitly applied within the extensions,
* rather than registered in registerconverters.cpp.
*
* $Id$
*
*****************************************************************************/

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <string>
#include <set>
#include <sstream>
#include <iostream>

#include <boost/python.hpp>

namespace bp = boost::python;

// This function will help convert a class Print statement into a __str__
// statement. See refinableobjclock_ext.cpp for an example.
template <class T>
std::string __str__(const T &obj)
{
    // Switch the stream buffer with std::cout, which is used by Print.
    ostringstream outbuf;
    streambuf* cout_strbuf(cout.rdbuf());
    std::cout.rdbuf(outbuf.rdbuf());
    // Call Print()
    obj.Print();
    // Switch the stream buffer back
    cout.rdbuf(cout_strbuf);

    string outstr = outbuf.str();
    // Remove the trailing newline
    size_t idx = outstr.find_last_not_of("\n");
    if (idx != string::npos)
        outstr.erase(idx+1);

    return outstr;
}

template <class T>
std::set<T> pyListToSet(const bp::list& l)
{

    std::set<T> cl;
    T typeobj;

    for(size_t i=0; i < len(l); ++i)
    {
        typeobj = bp::extract<T>(l[i]);
        cl.insert(typeobj);
    }

    return cl;
}

// For turning vector-like containers into lists
template <class T>
bp::list containerToPyList(T& v)
{
    bp::list l;

    for(typename T::iterator it = v.begin(); it != v.end(); ++it)
    {
        l.append(bp::object(*it));
    }
    return l;
}

template <class T>
bp::list setToPyList(std::set<T>& v)
{
    return containerToPyList< typename std::set<T> >(v);
}

#endif
