#!/bin/bash

# This is an installation script for a conda environment with pyobjcryst,
# also including ipython, jupyter notebook and ipywidgets.
# This has been tested on debian and macOS computers.
# It assumes you already have installed conda and compilers on your computer
# This will also download the libobjcryst and pyobjcryst repositories in the current directory

echo $1

if [ -z $2 ];
then
  echo "No directory or python executable given for installation !";
  echo "Usage: install-conda-env.sh ENVNAME PYTHON_VERSION"
  echo "   with: ENVNAME the name of the python virtual environement, e.g. pyobjcryst"
  echo "         PYTHON_VERSION the python version, e.g. 3.7"
  echo "example: install-conda-env.sh pyobjcryst 3.7"
  exit
fi

echo
echo "#############################################################################################"
echo " Creating conda environment"
echo "#############################################################################################"
echo

# create the conda virtual environment with necessary packages
conda create --yes -n $1 python=$2 pip
if [ $? -ne 0 ];
then
  echo "Conda environment creation failed."
  echo $?
  exit 1
fi

# Activate conda environment (see https://github.com/conda/conda/issues/7980)
eval "$(conda shell.bash hook)"
conda activate $1
if [ $? -ne 0 ];
then
  echo "Conda environment activation failed. Maybe 'conda init' is needed (see messages) ?"
  exit 1
fi

echo
echo "#############################################################################################"
echo " Adding required packages"
echo "#############################################################################################"
echo

conda install --yes -n $1 numpy matplotlib ipython notebook ipywidgets boost boost-cpp git scons

echo
echo "#############################################################################################"
echo " Installing libobjcryst from source"
echo "#############################################################################################"
echo

conda activate $1
git clone https://github.com/vincefn/libobjcryst.git
cd libobjcryst
# Why are the $CONDA_PREFIX include and lib directories  not automatically recognised ?
CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include scons prefix=$CONDA_PREFIX -j4 install
cd ..

echo
echo "#############################################################################################"
echo " Installing pyobjcryst"
echo "#############################################################################################"
echo

git clone https://github.com/vincefn/pyobjcryst.git
cd pyobjcryst
CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib scons prefix=$CONDA_PREFIX -j4 install
cd ..
