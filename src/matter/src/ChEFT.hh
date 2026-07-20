#ifndef ChEFT_hh
#define ChEFT_hh 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include <cmath>
#include "Constants.hh"
#include "ModelSpace.hh"

namespace ChEFT {
  //const double gA = 1.28899999999999992; // for check
  //const double delta_split = Constants::M_DELTA - 938.918267117927371; // for check
  const double hA = Constants::hA; // nucleon-delta axial g factor
  const double gA = 1.29; // Goldberger-Treiman
  const double fpi = Constants::F_PI;
  const double fpi2 = fpi * fpi;
  const double fpi4 = fpi2 * fpi2;
  const double pi = Constants::PI;
  const double hc = Constants::HBARC;
  const double hc2 = hc * hc;
  const double mpi = (2*Constants::M_PION_CHARGED + Constants::M_PION_NEUTRAL)/3;
  const double mpi2 = mpi * mpi;
  const double Lambda_chi = 700; // breakdown scale in MeV
  const double delta_split = Constants::M_DELTA - Constants::M_NUCLEON;
  const int nmesh_loop_integral = 25;

  const Eigen::Matrix<std::complex<double>, 2, 2> Id = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, 1.0).finished();
  const Eigen::Matrix<std::complex<double>, 2, 2> Px = (Eigen::Matrix2cd() << 0.0, 1.0, 1.0, 0.0).finished();
  const Eigen::Matrix<std::complex<double>, 2, 2> Py = (Eigen::Matrix2cd()
      << 0, -Constants::i_unit, Constants::i_unit, 0).finished();
  const Eigen::Matrix<std::complex<double>, 2, 2> Pz = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0,-1.0).finished();

  inline int sz2i(int sz) {return (sz==-1) ? 1 : 0;}; // spin up/down -> 0/1
  inline int tz2i(int tz) {return (tz==-1) ? 0 : 1;}; // proton/neutron -> 0/1
                                               //
  inline Eigen::Vector<std::complex<double>, 3> Sigma(int z1, int z2)
  {Eigen::Vector<std::complex<double>, 3> res;
    res << Px(sz2i(z1),sz2i(z2)), Py(sz2i(z1),sz2i(z2)), Pz(sz2i(z1),sz2i(z2));
    return res;};

  inline Eigen::Vector<std::complex<double>, 3> Tau(int z1, int z2)
  {Eigen::Vector<std::complex<double>, 3> res;
    res << Px(tz2i(z1),tz2i(z2)), Py(tz2i(z1),tz2i(z2)), Pz(tz2i(z1),tz2i(z2));
    return res;};

  class ChiralInteraction
  {
    public:
      ModelSpace* modelspace;
      int ch_order = 0;
      double Ct1S0_pp, Ct1S0_np, Ct1S0_nn, Ct3S1; // LO contact LECs in unit of MeV-2
      double C1S0, C3S1, C1P1, C3P0, C3P1, C3SD1, C3P2; // NLO contact LECs in unit of MeV-4
      double c1, c2, c3, c4; // N2LO 2pi LECs in unit of MeV-1
      double cD, cE; // N2LO 3N LECs, dimensionless
      double Lambda_cut = 500; // in unit of MeV
      double Lambda_SFR = -1; // in unit of MeV
      int reg_power = 3;
      bool include_delta = false;
      std::array<double, 3> CS = {0,0,0};
      std::array<double, 3> CT = {0,0,0};
      std::array<double, 7> Cnlo = {0,0,0,0,0,0,0};

      ~ChiralInteraction();
      ChiralInteraction(ModelSpace&);
      inline std::complex<double> CalcNNAsymME(int p, int q, int r, int s) {return CalcNNME(p,q,r,s) - CalcNNME(p,q,s,r);};
      inline std::complex<double> Calc3NAsymME(int p, int q, int r, int s, int t, int u)
      {return  Calc3NME(p, q, r, s, t, u) - Calc3NME(p, q, r, t, s, u)
        + Calc3NME(p, q, r, t, u, s) - Calc3NME(p, q, r, u, t, s)
          + Calc3NME(p, q, r, u, s, t) - Calc3NME(p, q, r, s, u, t) ;};

      std::complex<double> CalcNNME(int, int, int, int);
      std::complex<double> Calc3NME(int, int, int, int, int, int);

      void SetChiralOrder(int nu) {ch_order = nu;};
      void SetLECs(std::map<std::string,double>);
      void SetCt1S0_pp(double C) {Ct1S0_pp = C * 1.e-2;};
      void SetCt1S0_np(double C) {Ct1S0_np = C * 1.e-2;};
      void SetCt1S0_nn(double C) {Ct1S0_nn = C * 1.e-2;};
      void SetCt3S1(double C) {Ct3S1 = C * 1.e-2;};
      void SetC1S0(double C) {C1S0 = C * 1.e-8;};
      void SetC3S1(double C) {C3S1 = C * 1.e-8;};
      void SetC1P1(double C) {C1P1 = C * 1.e-8;};
      void SetC3P0(double C) {C3P0 = C * 1.e-8;};
      void SetC3P1(double C) {C3P1 = C * 1.e-8;};
      void SetC3SD1(double C) {C3SD1 = C * 1.e-8;};
      void SetC3P2(double C) {C3P2 = C * 1.e-8;};
      void Setc1(double C) {c1 = C * 1.e-3;};
      void Setc2(double C) {c2 = C * 1.e-3;};
      void Setc3(double C) {c3 = C * 1.e-3;};
      void Setc4(double C) {c4 = C * 1.e-3;};
      void SetcD(double C) {cD = C;};
      void SetcE(double C) {cE = C;};
      void SetDelta(bool delta) {include_delta = delta;};
      void SetLOContact();
      void SetNLOContact();
      void SetLambdaCut(double Lambda) {Lambda_cut = Lambda;};
      void SetLambdaSFR(double Lambda) {Lambda_SFR = Lambda;};
      void SetRegPower(double n) {reg_power = n;};

      std::complex<double> one_pion_exchange(SPState&, SPState&, SPState&, SPState&);
      std::complex<double> two_pion_exchange(SPState&, SPState&, SPState&, SPState&);
      std::complex<double> lo_contact(SPState&, SPState&, SPState&, SPState&);
      std::complex<double> nlo_contact(SPState&, SPState&, SPState&, SPState&);
      std::complex<double> two_pion_exchange_3n(SPState&, SPState&, SPState&, SPState&, SPState&, SPState&);
      std::complex<double> one_pion_exchange_3n(SPState&, SPState&, SPState&, SPState&, SPState&, SPState&);
      std::complex<double> contact_3n(SPState&, SPState&, SPState&, SPState&, SPState&, SPState&);

    private:
      double calc_loop_L(double q);
      double calc_loop_w(double q) {return std::sqrt(q*q + 4.0 * mpi*mpi);};
      double calc_loop_s() {return std::sqrt(Lambda_SFR*Lambda_SFR - 4.0 * mpi*mpi);};
      double calc_loop_A(double q);
      double calc_loop_H(double q);
      double calc_loop_D(double q);

      double DeltaTab(int i1, int i2) {return (i1==i2) ? 1.0 : 0.0;};
      std::complex<double> TauDotTau(SPState& p, SPState& q, SPState& r, SPState& s) {return Tau(p.tz,r.tz).transpose() * Tau(q.tz,s.tz);};
      std::complex<double> SigDotSig(SPState& p, SPState& q, SPState& r, SPState& s) {return Sigma(p.sz,r.sz).transpose() * Sigma(q.sz,s.sz);};

  };
}
#endif

