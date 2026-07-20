#pragma once
#include <complex>

#pragma omp declare reduction ( \
    + : std::complex<double> : omp_out += omp_in ) \
    initializer(omp_priv = std::complex<double>(0.0, 0.0))

