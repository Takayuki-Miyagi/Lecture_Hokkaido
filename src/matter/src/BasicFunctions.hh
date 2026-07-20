#ifndef BasicFuncions_hh
#define BasicFuncions_hh 1
#include <cmath>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include "Constants.hh"

namespace BasicFunctions{
  double CG(int ja, int ma, int jb, int mb, int J, int M);
  double ThreeJ(int j1, int j2, int j3, int m1, int m2, int m3);
  std::complex<double> Ylm(int, int, double, double);
}

#endif
