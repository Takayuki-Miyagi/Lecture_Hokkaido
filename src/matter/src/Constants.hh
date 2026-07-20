#ifndef Constants_h
#define Constants_h 1

#include <cmath>
#include <complex>
namespace Constants {
//  const double HBARC              = 197.326968;             // for check
//  const double M_PION_CHARGED     = 139.570179999999993;    // for chceck
//  const double M_PION_NEUTRAL     = 134.976599999999991;    // for check
//  const double M_NUCLEON          = 939.565;                // for check
                                                 //
                                                 //
  const double HBARC              = 197.3269718  ;                  // reduced Planck constant * speed of light in MeV * fm
  const double M_PROTON           = 938.2720813;                  // proton mass in MeV/c^2
  const double M_NEUTRON          = 939.5654133;                  // neutron mass in MeV/c^2
  const double M_ELECTRON         = 0.5109989461;                  // electron mass in MeV/c^2
  const double M_DELTA            = 1232.0;                        // delta isobar mass in MeV/c^2
  const double M_NUCLEON          = (M_PROTON+M_NEUTRON) / 2.0 ;    // avarage nucleon mass in MeV/c^2
  const double M_PION_CHARGED     = 139.57018 ;                     // charged pion mass in MeV/c^2
  const double M_PION_NEUTRAL     = 134.9766 ;                      // neutron pion mass in MeV/c^2


  const double gV = 1.00 ;                                          // nucleon vector g factor, which is 1 due to CVC
  const double gA = 1.27 ;                                          // nucleon axial g factor, aka gA. PDG lists 1.2723(23). From Lattice 1.271(13)
  const double hA = 1.40 ;                                          // Dimensionless delta-N axial coupling
  const double PROTON_SPIN_G      =  5.585690569;                   // proton spin g factor for magnetic moment from PDG 2020
  const double NEUTRON_SPIN_G     = -3.826085 ;                     // neutron spin g factor for magnetic moment from PDG 2020
  const double ELECTRON_SPIN_G    = 2.002319;                       // electron spin g factor
  const double ALPHA_FS           = 1.0 / 137.035999;               // fine structure constant
  const double F_PI               = 92.2 ;                          // pion decay constant

  // Math constants
  const double SQRT2    = sqrt(2.0);
  const double INVSQRT2 = 1.0 / SQRT2;
  const double PI       = 4.0*atan(1.0) ;
  const double SQRTPI   = sqrt(PI) ;
  const std::complex<double> i_unit = (std::complex<double>){0.0,1.0};

}
#endif
