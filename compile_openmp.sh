#!/bin/zsh
VERSION=1.1
CXX="$(brew --prefix llvm)/bin/clang++"

$CXX -O3 -march=native -std=c++17 \
  -Xpreprocessor -fopenmp \
  -I"$(brew --prefix libomp)/include" \
  -L"$(brew --prefix libomp)/lib" -lomp \
  ising_site_decorated_PT_v${VERSION}.cpp \
  -o ising_site_decorated_openmp_v${VERSION}

#export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
#OMP_NUM_THREADS=4 ./ising
