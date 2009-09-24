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
* $Id$
*
*****************************************************************************/

#include <iostream>
#include "helpers.hpp"

namespace bp = boost::python;

using namespace std;

void swapstdout(std::ostream& buf)
{
    // Switch the stream buffer with std::cout, which is used by Print.
    std::streambuf* cout_strbuf(std::cout.rdbuf());
    std::cout.rdbuf(buf.rdbuf());
    buf.rdbuf(cout_strbuf);
}
