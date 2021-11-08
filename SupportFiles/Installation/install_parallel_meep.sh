#!/bin/bash

# /usr/bin/pyvenv/pmp/lib # For Python
# /usr/local/lib/mpi/bin <3 For MPI
# ./configure --prefix=/usr/local/lib/pmeep # Where I would like to have installed meep

set -e

RPATH_FLAGS="-Wl,-rpath,/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/bin/pyvenv/pmp/lib:/usr/local/lib/mpi/lib"
MY_LDFLAGS="-L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/usr/bin/pyvenv/pmp/lib -L/usr/local/lib/mpi/lib ${RPATH_FLAGS}"
MY_CPPFLAGS="-I/usr/local/include -I/usr/include/hdf5/openmpi -I/usr/bin/pyvenv/pmp/include -I/usr/local/lib/mpi/include"

sudo apt-get update

# If building on Ubuntu 18.04LTS, replace libpng16-dev with libpng-dev,
# and libpython3.5-dev with libpython3-dev.
sudo apt-get -y install build-essential
sudo apt-get -y install gfortran
sudo apt-get -y install libblas-dev
sudo apt-get -y install liblapack-dev
sudo apt-get -y install libgmp-dev
sudo apt-get -y install swig
sudo apt-get -y install libgsl-dev
sudo apt-get -y install autoconf
sudo apt-get -y install pkg-config
sudo apt-get -y install libpng-dev  # libpng16-dev            \
# sudo apt-get -y install git
sudo apt-get -y install guile-3.0-dev # guile-2.0-dev           \
sudo apt-get -y install libfftw3-dev
sudo apt-get -y install libhdf5-openmpi-dev
sudo apt-get -y install hdf5-tools
sudo apt-get -y install libpython3-dev # libpython3.5-dev        \
sudo apt-get -y install python3-pip
sudo apt-get -y install cmake

#mkdir -p ~/Documents/Thesis/ThesisInstall

# cd ~/install
# git clone https://github.com/NanoComp/harminv.git
# cd harminv/
# sh autogen.sh --enable-shared
# make && sudo make install

cd ~/Documents/Thesis/ThesisInstall
git clone https://github.com/NanoComp/libctl.git
cd libctl/
sh autogen.sh --enable-shared
make && sudo make install

cd ~/Documents/Thesis/ThesisInstall
git clone https://github.com/NanoComp/h5utils.git
cd h5utils/
sh autogen.sh CC=mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
make && sudo make install

# cd ~/Documents/Thesis/ThesisInstall
# git clone https://github.com/NanoComp/mpb.git
# cd mpb/
# sh autogen.sh --enable-shared CC=mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}" --with-hermitian-eps
# make && sudo make install

cd ~/Documents/Thesis/ThesisInstall
git clone https://github.com/HomerReid/libGDSII.git
cd libGDSII/
sh autogen.sh
make && sudo make install

# The next line is only required on Ubuntu  16.04
# sudo pip3 install --upgrade pip

pip3 install --user --no-cache-dir mpi4py
pip3 install --user Cython==0.29.16
export HDF5_MPI="ON"
pip3 install --user --no-binary=h5py h5py
pip3 install --user autograd
pip3 install --user scipy
pip3 install --user matplotlib>3.0.0
pip3 install --user ffmpeg

cd ~/Documents/Thesis/ThesisInstall
git clone git://github.com/stevengj/nlopt.git
cd nlopt/
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 && make && sudo make install

cd ~/Documents/Thesis/ThesisInstall
# git clone https://github.com/NanoComp/meep.git --branch "v1.20.0" # had already run this line
cd meep/
# Open autogen and change line to ./configure --enable-maintainer-mode "$@" --prefix=/usr/bin/pyvenv/pmp
sh autogen.sh --enable-shared --with-mpi --with-openmp PYTHON=python3 LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
make && sudo make install
