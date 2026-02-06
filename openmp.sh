#!/bin/zsh
#export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
VERSION=1.1
echo "VERSION=" $VERSION | tee ising.log
date | tee -a ising.log 
OMP_NUM_THREADS=4 ./ising_site_decorated_openmp_v${VERSION} 2>&1 | tee -a ising.log &
wait
date | tee -a ising.log 
