#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_integration.h>
#include "ChEFT.hh"

namespace ChEFT {
  ChiralInteraction::~ChiralInteraction()
  {}

  ChiralInteraction::ChiralInteraction(ModelSpace& ms):
    modelspace(&ms)
  {}

  std::complex<double> ChiralInteraction::CalcNNME(int ip, int iq, int ir, int is)
  {
    SPState& p = modelspace->GetSPState(ip);
    SPState& q = modelspace->GetSPState(iq);
    SPState& r = modelspace->GetSPState(ir);
    SPState& s = modelspace->GetSPState(is);

    std::complex<double> res = {0.0, 0.0};
    res = one_pion_exchange(p, q, r, s);
    res+= two_pion_exchange(p, q, r, s);
    res+= lo_contact(p, q, r, s);
    res+= nlo_contact(p, q, r, s);

    double p2bra = modelspace->Get2BRelMomentum(p, q).squaredNorm() * hc2; // p'^2
    double p2ket = modelspace->Get2BRelMomentum(r, s).squaredNorm() * hc2; // p^2
    res *= exp(-std::pow(p2bra / (Lambda_cut*Lambda_cut), reg_power)) * exp(-std::pow(p2ket / (Lambda_cut*Lambda_cut), reg_power));
    res *= std::pow(hc / modelspace->Lbox, 3);
    res *= Constants::M_NUCLEON /
      std::sqrt(std::sqrt((Constants::M_NUCLEON * Constants::M_NUCLEON + p2bra) *
            (Constants::M_NUCLEON * Constants::M_NUCLEON + p2ket))); //relativity factor
    return res;
  }

  std::complex<double> ChiralInteraction::Calc3NME(int ip, int iq, int ir, int is, int it, int iu)
  {
    SPState& p = modelspace->GetSPState(ip);
    SPState& q = modelspace->GetSPState(iq);
    SPState& r = modelspace->GetSPState(ir);
    SPState& s = modelspace->GetSPState(is);
    SPState& t = modelspace->GetSPState(it);
    SPState& u = modelspace->GetSPState(iu);

    std::complex<double> res = {0.0, 0.0};
    res = two_pion_exchange_3n(p, q, r, s, t, u);
    res+= one_pion_exchange_3n(p, q, r, s, t, u);
    res+= contact_3n(p, q, r, s, t, u);

    double p2bra = modelspace->Get2BRelMomentum(p, q).squaredNorm() * hc2; // p'^2
    double p2ket = modelspace->Get2BRelMomentum(s, t).squaredNorm() * hc2; // p^2
    double q2bra = modelspace->Get3BJacMomentumQ(p, q, r).squaredNorm() * hc2; // q'^2
    double q2ket = modelspace->Get3BJacMomentumQ(s, t, u).squaredNorm() * hc2; // q^2
    res *= exp(-std::pow((p2bra+0.75*q2bra) / (Lambda_cut*Lambda_cut), reg_power)) * exp(-std::pow((p2ket+0.75*q2ket) / (Lambda_cut*Lambda_cut), reg_power));
    res *= std::pow(hc / modelspace->Lbox, 6);

    return res;
  }

  std::complex<double> ChiralInteraction::lo_contact(SPState& p, SPState& q, SPState& r, SPState& s)
  {
    int iT = (p.tz+q.tz)/2 + 1; // pp: 0, pn: 1, nn: 2
    std::complex<double> s_dot_s = Sigma(p.sz,r.sz).transpose() * Sigma(q.sz, s.sz); // sigma1 . sigma2
    std::complex<double> res;
    res = CS[iT] * DeltaTab(p.sz,r.sz) * DeltaTab(q.sz,s.sz) * DeltaTab(p.tz,r.tz) * DeltaTab(q.tz,s.tz);
    res+= CT[iT] * s_dot_s * DeltaTab(p.tz,r.tz) * DeltaTab(q.tz,s.tz);
    return res;
  }

  std::complex<double> ChiralInteraction::nlo_contact(SPState& p, SPState& q, SPState& r, SPState& s)
  {
    std::complex<double> res = {0.0, 0.0};
    if(ch_order < 1) return res;
    std::complex<double> isospin_delta = Id(tz2i(p.tz), tz2i(r.tz)) * Id(tz2i(q.tz), tz2i(s.tz));
    if(std::abs(isospin_delta) < 1.e-16) return res;

    Eigen::Vector<double, 3> qmom = (modelspace->Get2BRelMomentum(p, q) - modelspace->Get2BRelMomentum(r, s)) * hc; // p' - p
    Eigen::Vector<double, 3> kmom = (modelspace->Get2BRelMomentum(p, q) + modelspace->Get2BRelMomentum(r, s)) * hc * 0.5; // (p' + p) / 2
    Eigen::Vector<std::complex<double>, 3> Svec = (Sigma(p.sz,r.sz) * Id(sz2i(q.sz), sz2i(s.sz)) + Sigma(q.sz,s.sz) * Id(sz2i(p.sz), sz2i(r.sz))) * 0.5; // (sigma1 + sigma2) / 2
    std::complex<double> s_dot_s = SigDotSig(p,q,r,s); // sigma1 . sigma2

    double q2 = qmom.squaredNorm();
    double k2 = kmom.squaredNorm();
    std::complex<double> S_dot_q_x_k = Svec.transpose() * qmom.cross(kmom);
    std::complex<double> spin_delta = DeltaTab(p.sz,r.sz) * DeltaTab(q.sz,s.sz);

    res = Cnlo[0] * q2 * spin_delta;
    res+= Cnlo[1] * k2 * spin_delta;
    res+= Cnlo[2] * q2 * s_dot_s;
    res+= Cnlo[3] * k2 * s_dot_s;
    res+= Cnlo[4] * (-Constants::i_unit * S_dot_q_x_k);
    res+= Cnlo[5] * (qmom.dot(Sigma(p.sz, r.sz)) * qmom.dot(Sigma(q.sz, s.sz)));
    res+= Cnlo[6] * (kmom.dot(Sigma(p.sz, r.sz)) * kmom.dot(Sigma(q.sz, s.sz)));
    return res * isospin_delta;
  }

  std::complex<double> ChiralInteraction::one_pion_exchange(SPState& p, SPState& q, SPState& r, SPState& s)
  {
    std::complex<double> res = {0.0, 0.0};
    Eigen::Vector<double, 3> qmom = (modelspace->Get2BRelMomentum(p, q) - modelspace->Get2BRelMomentum(r, s)) * hc; // p' - p
    double q2 = qmom.squaredNorm();
    double fact = - gA*gA / (4.0 * fpi2);
    double isospin_factor = 0.0;
    if(ch_order == 0) {
      // CIB effect is not included
      std::complex<double> tau_dot_tau = TauDotTau(p, q, r, s);
      res = fact * tau_dot_tau * (qmom.dot(Sigma(p.sz, r.sz)) * qmom.dot(Sigma(q.sz, s.sz))) / (q2 + mpi2);
    }
    else{
      // CIB effect is included
      if(p.tz + q.tz != 0){
        res = fact * (qmom.dot(Sigma(p.sz, r.sz)) * qmom.dot(Sigma(q.sz, s.sz))) / (q2 + Constants::M_PION_NEUTRAL*Constants::M_PION_NEUTRAL);
      }
      else{
        isospin_factor = gsl_sf_coupling_3j(1, 1, 0, p.tz, q.tz, 0) * gsl_sf_coupling_3j(1, 1, 0, r.tz, s.tz, 0);
        res = fact * isospin_factor * (qmom.dot(Sigma(p.sz, r.sz)) * qmom.dot(Sigma(q.sz, s.sz))) * (-1.0 / (q2 + Constants::M_PION_NEUTRAL*Constants::M_PION_NEUTRAL) - 2.0 / (q2 + Constants::M_PION_CHARGED*Constants::M_PION_CHARGED));
        isospin_factor = gsl_sf_coupling_3j(1, 1, 2, p.tz, q.tz, 0) * gsl_sf_coupling_3j(1, 1, 2, r.tz, s.tz, 0) * 3.0;
        res+= fact * isospin_factor * (qmom.dot(Sigma(p.sz, r.sz)) * qmom.dot(Sigma(q.sz, s.sz))) * (-1.0 / (q2 + Constants::M_PION_NEUTRAL*Constants::M_PION_NEUTRAL) + 2.0 / (q2 + Constants::M_PION_CHARGED*Constants::M_PION_CHARGED));
      }
    }
    return res;
  }

  std::complex<double> ChiralInteraction::two_pion_exchange(SPState& p, SPState& q, SPState& r, SPState& s)
  {
    std::complex<double> res = {0.0, 0.0};
    if(ch_order == 0) return res; //LO

    Eigen::Vector<double, 3> qmom = (modelspace->Get2BRelMomentum(p, q) - modelspace->Get2BRelMomentum(r, s)) * hc; // p' - p
    Eigen::Vector<double, 3> kmom = (modelspace->Get2BRelMomentum(p, q) + modelspace->Get2BRelMomentum(r, s)) * hc / 2.0; // (p' + p) / 2
    Eigen::Vector<std::complex<double>, 3> Svec = (Sigma(p.sz,r.sz) + Sigma(q.sz,s.sz)) / 2; // (sigma1 + sigma2) / 2

    double q2 = qmom.squaredNorm();
    double qnorm = std::sqrt(q2); //|qmom|
    double mpi2 = mpi * mpi;
    double mpi4 = mpi2 * mpi2;
    double mpi5 = mpi * mpi4;
    double gA2 = gA * gA;
    double gA4 = gA2 * gA2;
    double hA2 = hA * hA;
    double hA4 = hA2 * hA2;
    double w = calc_loop_w(qnorm);
    double w_tilde = std::sqrt(2.0 * mpi2 + q2);
    double Lambda_SFR2 = Lambda_SFR * Lambda_SFR;
    double Delta = delta_split;
    double Delta2 = Delta*Delta;
    double SIGMA = 2.0 * mpi2 + q2 - 2.0 * Delta2;
    double L_q = calc_loop_L(qnorm);
    double A_q = calc_loop_A(qnorm);
    double loop_H = 0.0;
    double loop_D = 0.0;
    double isospin_delta = DeltaTab(p.tz,r.tz) * DeltaTab(q.tz,s.tz);
    double spin_delta = DeltaTab(p.sz,r.sz) * DeltaTab(q.sz,s.sz);
    std::complex<double> tau_dot_tau = TauDotTau(p, q, r, s);
    std::complex<double> sig_dot_sig = SigDotSig(p, q, r, s);
    std::complex<double> S_dot_q_x_k = Svec.transpose() * qmom.cross(kmom);

    if(include_delta){
      loop_H = calc_loop_H(qnorm);
      loop_D = calc_loop_D(qnorm);
    }

    double V_C = 0.0, W_C = 0.0, V_S = 0.0, W_S = 0.0, V_LS = 0.0, W_LS = 0.0, V_T = 0.0, W_T = 0.0;
    if(ch_order >= 1){
      W_C += -L_q / (384.0 * Constants::PI*Constants::PI * fpi4) *
        (4.0*mpi2*(5.0*gA4-4.0*gA2-1.0) + q2*(23.0*gA4-10.0*gA2-1.0) + 48.0*gA4*mpi4/(w*w));
      V_T += -3.0 * gA4 * L_q / (64.0 * Constants::PI * Constants::PI * fpi4);
      V_S += q2 * 3.0 * gA4 * L_q / (64.0 * Constants::PI * Constants::PI * fpi4);
    }
    if(include_delta and ch_order >= 1){

      // delta single triangle diagrams:
      W_C += (- hA2) / (216.0 * Constants::PI * Constants::PI * fpi4) * ((6.0 * SIGMA - w*w)*L_q + 12.0*Delta2*SIGMA*loop_D);

      // delta leading single box diagrams:
      V_C += - gA2 * hA2 / (12.0*Constants::PI*fpi4*Delta) * (2.0*mpi2 + q2) * (2.0*mpi2 + q2) * A_q;
      W_C += - gA2 * hA2 / (216.0*Constants::PI*Constants::PI*fpi4) * ((12.0*Delta2-20.0*mpi2-11.0*q2)*L_q + 6.0*SIGMA*SIGMA*loop_D);
      V_T += - gA2 * hA2 / (48.0*Constants::PI*Constants::PI*fpi4) * (-2.0*L_q + (w*w-4.0*Delta2)*loop_D);
      V_S += q2 * gA2 * hA2 / (48.0*Constants::PI*Constants::PI*fpi4) * (-2.0*L_q + (w*w-4.0*Delta2)*loop_D);
      W_T += - gA2 * hA2 / (144.0*Constants::PI*fpi4*Delta) * w*w*A_q;
      W_S += q2 * gA2 * hA2 / (144.0*Constants::PI*fpi4*Delta) * w*w*A_q;

      // delta leading double box diagrams:
      V_C += - hA4 / (27.0*Constants::PI*Constants::PI*fpi4) * (-4.0*Delta2*L_q + SIGMA*(loop_H + (SIGMA+8.0*Delta2)*loop_D));
      W_C += - hA4 / (486.0*Constants::PI*Constants::PI*fpi4) * ((12.0*SIGMA-w*w)*L_q + 3.0*SIGMA*(loop_H+(8.0*Delta2-SIGMA)*loop_D));
      V_T += - hA4 / (216.0*Constants::PI*Constants::PI*fpi4) * (6.0*L_q + (12.0*Delta2-w*w)*loop_D);
      V_S += q2 * hA4 / (216.0*Constants::PI*Constants::PI*fpi4) * (6.0*L_q + (12.0*Delta2-w*w)*loop_D);
      W_T += - hA4 / (1296.0*Constants::PI*Constants::PI*fpi4) * (2.0*L_q + (4.0*Delta2+w*w)*loop_D);
      W_S += q2 * hA4 / (1296.0*Constants::PI*Constants::PI*fpi4) * (2.0*L_q + (4.0*Delta2+w*w)*loop_D);
    }

    if(ch_order >= 2){
      // LEC dependent terms
      V_C += 3.0 * gA2 / (16.0 * Constants::PI * fpi4) * (- (2.0 * mpi2 * (2.0 * c1 - c3) - q2 * c3) * w_tilde * w_tilde * A_q);
      W_T += -gA2 * A_q / (32.0 * Constants::PI * fpi4) * (c4 * w * w);
      W_S += -q2 * -gA2 * A_q / (32.0 * Constants::PI * fpi4) * (c4 * w * w);
      if(not include_delta){
        // relativistic correction (1/m dependent)
        V_C += 3.0 * gA2 / (16.0 * Constants::PI * fpi4) * (gA2 * mpi5 / (16.0 * Constants::M_NUCLEON * w * w) + 3.0*q2*gA2 / (16.0*Constants::M_NUCLEON)*w_tilde*w_tilde*A_q);
        W_C += gA2 / (128.0 * Constants::PI * Constants::M_NUCLEON * fpi4) *
          (3.0*gA2*mpi5/(w*w) - (4.0*mpi2 + 2.0*q2 - gA2*(4.0*mpi2+3.0*q2)) * w_tilde*w_tilde*A_q);
        V_T += 9.0*gA4*w_tilde*w_tilde*A_q / (512.0*Constants::PI*Constants::M_NUCLEON*fpi4);
        W_T += -gA2*A_q / (32.0 * Constants::PI * fpi4) * (w*w/(4.0*Constants::M_NUCLEON) - gA2/(8.0*Constants::M_NUCLEON) * (10.0*mpi2 + 3.0*q2));
        V_S += -q2 * 9.0*gA4*w_tilde*w_tilde*A_q / (512.0*Constants::PI*Constants::M_NUCLEON*fpi4);
        W_S += q2*gA2*A_q / (32.0 * Constants::PI * fpi4) * (w*w/(4.0*Constants::M_NUCLEON) - gA2/(8.0*Constants::M_NUCLEON) * (10.0*mpi2 + 3.0*q2));
        V_LS += 3.0*gA4*w_tilde*w_tilde*A_q / (32.0*Constants::PI*Constants::M_NUCLEON*fpi4);
        W_LS += gA2*(1.0 - gA2)*w*w*A_q / (32.0*Constants::PI*Constants::M_NUCLEON*fpi4);
      }
    }
    if(include_delta and ch_order >= 2){
      // delta subleading triangle diagrams
      V_C += - hA2*Delta / (18.0*Constants::PI*Constants::PI*fpi4) * (6.0*SIGMA*(4.0*c1*mpi2-2.0*c2*Delta2-c3*(2*Delta2+SIGMA))*loop_D
          + (-24.0*c1*mpi2+c2*(w*w-6.0*SIGMA)+6.0*c3*(2.0*Delta2+SIGMA))*L_q);
      W_T += - c4*hA2*Delta / (72.0*Constants::PI*Constants::PI*fpi4) * ((w*w-4.0*Delta2)*loop_D - 2.0*L_q);
      W_S += q2 * c4*hA2*Delta / (72.0*Constants::PI*Constants::PI*fpi4) * ((w*w-4.0*Delta2)*loop_D - 2.0*L_q);
      // other terms are proportinal to b3+b8, which can be renormalized.
    }
    res += ( V_C * isospin_delta +  W_C * tau_dot_tau) * spin_delta;
    res += ( V_S * isospin_delta +  W_S * tau_dot_tau) * sig_dot_sig;
    res += (V_LS * isospin_delta + W_LS * tau_dot_tau) * (-Constants::i_unit * S_dot_q_x_k);
    res += ( V_T * isospin_delta +  W_T * tau_dot_tau) * (qmom.dot(Sigma(p.sz, r.sz)) * qmom.dot(Sigma(q.sz, s.sz)));
    return res;
  }

  std::complex<double> ChiralInteraction::two_pion_exchange_3n(SPState& p, SPState& q, SPState& r, SPState& s, SPState& t, SPState& u)
  {
    std::complex<double> res = {0.0, 0.0};
    if(ch_order == 0) return res;
    double prefactor1 = gA*gA * mpi2 / fpi4;
    double prefactor3 = 0.5 * gA*gA / fpi4;
    double prefactor4 = 0.25 * gA*gA / fpi4;
    Eigen::Vector<double, 3> q1 = (modelspace->GetMomentum(p) - modelspace->GetMomentum(s)) * hc;
    Eigen::Vector<double, 3> q2 = (modelspace->GetMomentum(q) - modelspace->GetMomentum(t)) * hc;
    Eigen::Vector<double, 3> q3 = (modelspace->GetMomentum(r) - modelspace->GetMomentum(u)) * hc;
    std::complex<double> tau_dot_tau_12 = TauDotTau(p,q,s,t) * DeltaTab(r.tz,u.tz) * DeltaTab(r.sz,u.sz);
    std::complex<double> tau_dot_tau_23 = TauDotTau(q,r,t,u) * DeltaTab(p.tz,s.tz) * DeltaTab(p.sz,s.sz);
    std::complex<double> tau_dot_tau_13 = TauDotTau(p,r,s,u) * DeltaTab(q.tz,t.tz) * DeltaTab(q.sz,t.sz);
    Eigen::Vector<std::complex<double>,3> tau1 = Tau(p.tz,s.tz);
    Eigen::Vector<std::complex<double>,3> tau2 = Tau(q.tz,t.tz);
    Eigen::Vector<std::complex<double>,3> tau3 = Tau(r.tz,u.tz);
    Eigen::Vector<std::complex<double>,3> tau1_x_tau2;
    tau1_x_tau2(0) = tau1(1) * tau2(2) - tau1(2) * tau2(1);
    tau1_x_tau2(1) = tau1(2) * tau2(0) - tau1(0) * tau2(2);
    tau1_x_tau2(2) = tau1(0) * tau2(1) - tau1(1) * tau2(0);
    std::complex<double> tau_dot_tau_x_tau = tau3.transpose() * tau1_x_tau2;

    std::complex<double> pe_12, func1_12, func3_12, func4_12;
    std::complex<double> pe_23, func1_23, func3_23, func4_23;
    std::complex<double> pe_13, func1_13, func3_13, func4_13;

    pe_12 = q1.dot(Sigma(p.sz,s.sz)) * q2.dot(Sigma(q.sz,t.sz)) / ((q1.squaredNorm() + mpi2) * (q2.squaredNorm() + mpi2));
    pe_23 = q2.dot(Sigma(q.sz,t.sz)) * q3.dot(Sigma(r.sz,u.sz)) / ((q2.squaredNorm() + mpi2) * (q3.squaredNorm() + mpi2));
    pe_13 = q1.dot(Sigma(p.sz,s.sz)) * q3.dot(Sigma(r.sz,u.sz)) / ((q1.squaredNorm() + mpi2) * (q3.squaredNorm() + mpi2));

    func1_12 = pe_12;
    func1_23 = pe_23;
    func1_13 = pe_13;

    func3_12 = q1.dot(q2) * pe_12;
    func3_23 = q2.dot(q3) * pe_23;
    func3_13 = q1.dot(q3) * pe_13;

    func4_12 = q1.cross(q2).dot(Sigma(r.sz,u.sz)) * pe_12;
    func4_23 = q2.cross(q3).dot(Sigma(p.sz,s.sz)) * pe_23;
    func4_13 = q3.cross(q1).dot(Sigma(q.sz,t.sz)) * pe_13;

    if(include_delta) {
      double c_delta = -4.0 * hA*hA / (9.0 * delta_split);
      res += (c_delta * prefactor3 * func3_12) * tau_dot_tau_12 + (-0.5 * c_delta) * prefactor4 * func4_12 * tau_dot_tau_x_tau;
      res += (c_delta * prefactor3 * func3_23) * tau_dot_tau_23 + (-0.5 * c_delta) * prefactor4 * func4_23 * tau_dot_tau_x_tau;
      res += (c_delta * prefactor3 * func3_13) * tau_dot_tau_13 + (-0.5 * c_delta) * prefactor4 * func4_13 * tau_dot_tau_x_tau;
    }

    if(ch_order >= 2){
      res += (-c1*prefactor1 * func1_12 + c3*prefactor3 * func3_12) * tau_dot_tau_12 + c4*prefactor4 * func4_12 * tau_dot_tau_x_tau;
      res += (-c1*prefactor1 * func1_23 + c3*prefactor3 * func3_23) * tau_dot_tau_23 + c4*prefactor4 * func4_23 * tau_dot_tau_x_tau;
      res += (-c1*prefactor1 * func1_13 + c3*prefactor3 * func3_13) * tau_dot_tau_13 + c4*prefactor4 * func4_13 * tau_dot_tau_x_tau;
    }
    return res;
  }

  std::complex<double> ChiralInteraction::one_pion_exchange_3n(SPState& p, SPState& q, SPState& r, SPState& s, SPState& t, SPState& u)
  {
    std::complex<double> res = 0.0;
    if(ch_order < 2) return res;
    double prefactor = - gA * cD / (8.0 * fpi4 * Lambda_chi);
    Eigen::Vector<double, 3> q1 = (modelspace->GetMomentum(p) - modelspace->GetMomentum(s)) * hc;
    Eigen::Vector<double, 3> q2 = (modelspace->GetMomentum(q) - modelspace->GetMomentum(t)) * hc;
    Eigen::Vector<double, 3> q3 = (modelspace->GetMomentum(r) - modelspace->GetMomentum(u)) * hc;
    std::complex<double> tau_dot_tau_12 = TauDotTau(p,q,s,t) * DeltaTab(r.tz,u.tz) * DeltaTab(r.sz, u.sz);
    std::complex<double> tau_dot_tau_23 = TauDotTau(q,r,t,u) * DeltaTab(p.tz,s.tz) * DeltaTab(p.sz, s.sz);
    std::complex<double> tau_dot_tau_13 = TauDotTau(p,r,s,u) * DeltaTab(q.tz,t.tz) * DeltaTab(q.sz, t.sz);
    std::complex<double> func_mom = 0.0;

    func_mom = q2.dot(Sigma(q.sz,t.sz)) * q2.dot(Sigma(r.sz,u.sz)) / (q2.squaredNorm() + mpi2);
    func_mom+= q3.dot(Sigma(q.sz,t.sz)) * q3.dot(Sigma(r.sz,u.sz)) / (q3.squaredNorm() + mpi2);
    res = prefactor * func_mom * tau_dot_tau_23;

    func_mom = q1.dot(Sigma(p.sz,s.sz)) * q1.dot(Sigma(q.sz,t.sz)) / (q1.squaredNorm() + mpi2);
    func_mom+= q2.dot(Sigma(p.sz,s.sz)) * q2.dot(Sigma(q.sz,t.sz)) / (q2.squaredNorm() + mpi2);
    res+= prefactor * func_mom * tau_dot_tau_12;

    func_mom = q1.dot(Sigma(p.sz,s.sz)) * q1.dot(Sigma(r.sz,u.sz)) / (q1.squaredNorm() + mpi2);
    func_mom+= q3.dot(Sigma(p.sz,s.sz)) * q3.dot(Sigma(r.sz,u.sz)) / (q3.squaredNorm() + mpi2);
    res+= prefactor * func_mom * tau_dot_tau_13;
    return res;
  }

  std::complex<double> ChiralInteraction::contact_3n(SPState& p, SPState& q, SPState& r, SPState& s, SPState& t, SPState& u)
  {
    std::complex<double> res = 0.0;
    if(ch_order < 2) return res;
    double prefactor = cE / (fpi4 * Lambda_chi); // the factor 2 is from sum_{ij} --> sum_{i<j}
    double spin_delta = DeltaTab(p.sz,s.sz) * DeltaTab(q.sz,t.sz) * DeltaTab(r.sz, u.sz);
    std::complex<double> tau_dot_tau = TauDotTau(p,q,s,t) * DeltaTab(r.tz,u.tz);
    tau_dot_tau += TauDotTau(p,r,s,u) * DeltaTab(q.tz,t.tz);
    tau_dot_tau += TauDotTau(q,r,t,u) * DeltaTab(p.tz,s.tz);
    res = prefactor * tau_dot_tau * spin_delta;
    return res;
  }

  void ChiralInteraction::SetLECs(std::map<std::string,double> LECs)
  {
    for(auto& it : LECs){
      if(it.first == "Ct1S0_pp") SetCt1S0_pp(it.second);
      if(it.first == "Ct1S0_np") SetCt1S0_np(it.second);
      if(it.first == "Ct1S0_nn") SetCt1S0_nn(it.second);
      if(it.first == "Ct3S1") SetCt3S1(it.second);
      if(it.first == "C1S0") SetC1S0(it.second);
      if(it.first == "C3S1") SetC3S1(it.second);
      if(it.first == "C1P1") SetC1P1(it.second);
      if(it.first == "C3P0") SetC3P0(it.second);
      if(it.first == "C3P1") SetC3P1(it.second);
      if(it.first == "C3SD1") SetC3SD1(it.second);
      if(it.first == "C3P2") SetC3P2(it.second);
      if(it.first == "c1") Setc1(it.second);
      if(it.first == "c2") Setc2(it.second);
      if(it.first == "c3") Setc3(it.second);
      if(it.first == "c4") Setc4(it.second);
      if(it.first == "cD") SetcD(it.second);
      if(it.first == "cE") SetcE(it.second);
    }
    SetLOContact();
    SetNLOContact();
  }

  void ChiralInteraction::SetLOContact()
  {
    CS[0] = (3*Ct3S1 + Ct1S0_pp) / (16 * pi);
    CS[1] = (3*Ct3S1 + Ct1S0_np) / (16 * pi);
    CS[2] = (3*Ct3S1 + Ct1S0_nn) / (16 * pi);
    CT[0] = (Ct3S1 - Ct1S0_pp) / (16 * pi);
    CT[1] = (Ct3S1 - Ct1S0_np) / (16 * pi);
    CT[2] = (Ct3S1 - Ct1S0_nn) / (16 * pi);
  }

  void ChiralInteraction::SetNLOContact()
  {
    Eigen::Vector<double, 7> cpw;
    Eigen::Vector<double, 7> cop;
    cpw << C1S0, C3P0, C1P1, C3P1, C3S1, C3SD1, C3P2;
    Eigen::Matrix<double, 7, 7> TransMat;
    TransMat <<
           1.0,      0.25,      -3.0,    -0.75,        0.0,                   -1.0,              -0.25,
      -2.0/3.0,   1.0/6.0,  -2.0/3.0,  1.0/6.0,   -2.0/3.0,                    2.0,               -0.5,
      -2.0/3.0,   1.0/6.0,       2.0,     -0.5,        0.0,                2.0/3.0,           -1.0/6.0,
      -2.0/3.0,   1.0/6.0,  -2.0/3.0,  1.0/6.0,   -1.0/3.0,               -4.0/3.0,            1.0/3.0,
           1.0,      0.25,       1.0,     0.25,        0.0,                1.0/3.0,           1.0/12.0,
           0.0,       0.0,       0.0,      0.0,        0.0,  -2.0*std::sqrt(2)/3.0,  -std::sqrt(2)/6.0,
      -2.0/3.0,   1.0/6.0,  -2.0/3.0,  1.0/6.0,    1.0/3.0,                    0.0,                0.0;
    TransMat *= 4 * pi;
    cop = TransMat.inverse() * cpw;
    for(int i = 0; i < 7; i++){
      Cnlo[i] = cop(i);
    }
  }

  double ChiralInteraction::calc_loop_L(double q)
  {
    double res = 1.0; // q -> 0 limit (DR)
    double eps = 1.e-8;
    double loop_s = calc_loop_s();
    double loop_w = calc_loop_w(q);
    if(Lambda_SFR >= 2*mpi and q == 0.0){
      res = (0.5 * loop_w / eps) * std::log((Lambda_SFR*Lambda_SFR*loop_w*loop_w + eps*eps*loop_s*loop_s + 2.0*Lambda_SFR*eps*loop_w*loop_s) / (4.0*mpi2*(Lambda_SFR*Lambda_SFR+eps*eps)));
    }
    else {
      if(Lambda_SFR >= 2*mpi and q > 0.0){
        // spectral function regularization
        res = (0.5 * loop_w / q) * std::log((Lambda_SFR*Lambda_SFR*loop_w*loop_w + q*q*loop_s*loop_s + 2.0*Lambda_SFR*q*loop_w*loop_s) / (4.0*mpi2*(Lambda_SFR*Lambda_SFR+q*q)));
      }
      else {
        // dimensional regularization
        res = (loop_w / q) * std::log(0.5 * (loop_w + q) / mpi);
      }
    }
    return res;
  }

  double ChiralInteraction::calc_loop_A(double q)
  {
    double res = 0.25 / mpi; // q -> 0 limit (DR)
    if(Lambda_SFR >= 2*mpi and q == 0.0){
      double eps = 1.e-8;
      res = (0.5 / eps) * std::atan(eps*(Lambda_SFR-2.0*mpi) / (eps*eps + 2.0*Lambda_SFR*mpi));
    }
    else{
      if(Lambda_SFR >= 2*mpi){
        // spectral function regularization
        res = (0.5 / q) * std::atan(q*(Lambda_SFR-2.0*mpi) / (q*q + 2.0*Lambda_SFR*mpi));
      }
      else{
        // dimensional regularization
        res = (0.5 / q) * std::atan(0.5 * q / mpi);
      }
    }
    return res;
  }

  double ChiralInteraction::calc_loop_H(double q)
  {
    double Delta = delta_split;
    double Delta2 = Delta*Delta;
    double SIGMA = 2.0 * mpi2 + q*q - 2.0 * Delta2;
    double loop_w = calc_loop_w(q);
    return 2.0 * SIGMA / (loop_w*loop_w - 4.0*Delta2) * (calc_loop_L(q) - calc_loop_L(2.0 * std::sqrt(Delta2-mpi*mpi)));
  }

  double ChiralInteraction::calc_loop_D(double q)
  {
    double Delta = delta_split;
    gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(nmesh_loop_integral);
    double res = 0.0;
    for (int i=0; i<nmesh_loop_integral; i++){
      double mu, w;
      gsl_integration_glfixed_point(2.0*mpi, Lambda_SFR, i, &mu, &w, table);
      res += w * std::atan( std::sqrt(mu*mu - 4.0*mpi*mpi) / (2.0*Delta) ) / (mu*mu + q*q);
    }
    gsl_integration_glfixed_table_free(table);
    return res / Delta;
  }
}
