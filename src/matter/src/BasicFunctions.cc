#include "BasicFunctions.hh"

namespace BasicFunctions{
  /// Wigner 3j symbol
  double ThreeJ(int j1, int j2, int j3, int m1, int m2, int m3)
  {
    return gsl_sf_coupling_3j(j1, j2, j3, m1, m2, m3);
  }

  /// Clebsch-Gordan coefficient
  double CG(int ja, int ma, int jb, int mb, int J, int M)
  {
    return std::pow(-1.0, (ja-jb+M)/2) * std::sqrt(J+1) * ThreeJ(ja,jb,J,ma,mb,-M);
  }

  std::complex<double> Ylm(int l, int m, double theta, double phi){
    double Yl0 = gsl_sf_legendre_sphPlm(l,std::abs(m),std::cos(theta));
    return std::pow(-1.0, (m-std::abs(m))/2) * Yl0 * std::exp(Constants::i_unit * phi);
  }
}
