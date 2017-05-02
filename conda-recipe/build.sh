#!/bin/bash

export CPATH="${PREFIX}/include:$CPATH"
export LIBRARY_PATH="${PREFIX}/lib:$LIBRARY_PATH"
MYNCPU=$(( (CPU_COUNT > 4) ? 4 : CPU_COUNT ))

if [ `uname` == Darwin ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib:$DYLD_FALLBACK_LIBRARY_PATH"
fi

scons -j $MYNCPU install prefix=$PREFIX

# Add more build steps here, if they are necessary.

# See http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
