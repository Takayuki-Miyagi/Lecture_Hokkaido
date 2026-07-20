#include "IMSRG.hh"

//
// This is imported from imsrg++ by S.R.Stroberg
//
IMSRG::IMSRG(double smax, double ds_max, double omega_norm_max, double eta_criterion)
  : smax(smax), ds_max(ds_max), omega_norm_max(omega_norm_max), eta_criterion(eta_criterion)
{
  s = 0.0;
}

IMSRG::IMSRG(Operator& H_in, double smax, double ds_max, double omega_norm_max, double eta_criterion)
  : smax(smax), ds_max(ds_max), omega_norm_max(omega_norm_max), eta_criterion(eta_criterion)
{
  s = 0.0;
  Hin = &H_in;

  Hs = Operator(*Hin);
  Eta = Operator(*(Hs.modelspace));
  Eta.SetAntiSym(true);
  Omega_s.emplace_back(Eta);
  Omega_s.back().SetAntiSym(true);
}

void IMSRG::Solve(std::string method)
{
  if(method=="magnus") solve_magnus();
  if(method=="RK4") solve_runge_kutta4();
}

Operator IMSRG::BCHTransformation(const Operator& op, const Operator& Omega)
{
  double start_time = omp_get_wtime();

  int max_iter = 40;
  int warn_iter = 12;
  double norm_op = op.Norm();
  double norm_w = Omega.Norm();
  Operator OpOut = op;
  Operator OpNested = OpOut;
  double factorial_denom = 1.0;
  double epsilon = norm_op * exp(-2 * norm_w) * bch_transform_threshold / (2*norm_w);
  op.TwoBody.EnsureXOperator();
  Omega.TwoBody.EnsureXOperator();
  for(int i = 1; i <= max_iter; i++){
    OpNested = Commutator::Commutator(Omega, OpNested);
    factorial_denom /= (double)i;
    OpOut += OpNested * factorial_denom;
    epsilon *= (double)(i+1);
    if(OpNested.Norm() < epsilon) break;
    if(i==warn_iter) std::cout << "Warning: BCHTransformation not converged after " << warn_iter << " nested commutators" << std::endl;
    if(i==max_iter) std::cout << "Warning: BCHTransformation did not converge after " << max_iter << " nested commutators" << std::endl;
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
  return OpOut;
}

Operator IMSRG::Evolve(const Operator& op)
{
  Operator OpOut = op;
  for(int i = 0; i < Omega_s.size(); i++){
     OpOut = BCHTransformation(OpOut, Omega_s[i]);
  }
  return OpOut;
}

//*****************************************************************************************
// Baker-Campbell-Hausdorff formula
//  returns Z, where
//  exp(Z) = exp(X) * exp(Y).
//  Z = X + Y + 1/2[X, Y]
//     + 1/12 [X,[X,Y]] + 1/12 [Y,[Y,X]]
//     - 1/24 [Y,[X,[X,Y]]]
//     - 1/720 [Y,[Y,[Y,[Y,X]]]] - 1/720 [X,[X,[X,[X,Y]]]]
//     + ...
//*****************************************************************************************
/// X.BCH_Product(Y) returns \f$Z\f$ such that \f$ e^{Z} = e^{X}e^{Y}\f$
/// by employing the [Baker-Campbell-Hausdorff formula](http://en.wikipedia.org/wiki/Baker-Campbell-Hausdorff_formula)
/// \f[ Z = X + Y + \frac{1}{2}[X,Y] + \frac{1}{12}([X,[X,Y]]+[Y,[Y,X]]) + \ldots \f]
//*****************************************************************************************
Operator IMSRG::BCHProduct(const Operator& X, const Operator &Y)
{
  if(X.antisym != Y.antisym) {
    std::cout << "X and Y should be both symmetric or skew-symmetric!" << std::endl;
    exit(0);
  }
  double start_time = omp_get_wtime();

  double nx = X.Norm();
  double ny = Y.Norm();
  std::vector<double> bernoulli = {1.0, -0.5, 1. / 6, 0.0, -1. / 30, 0.0, 1. / 42, 0, -1. / 30};
  std::vector<double> factorial = {1.0, 1.0, 2.0, 6.0, 24., 120., 720., 5040., 40320.};

  Operator Z = X + Y;
  Z.SetAntiSym(X.antisym);
  Operator Nested = Commutator::Commutator(Y, X);
  double nxy = Nested.Norm();
  int k = 1;
  // We assume X is small, but just in case, we check if we should include the [X,[X,Y]] term.
  if (nxy * nx > bch_product_threshold)
  {
    Z += (1. / 12) * Commutator::Commutator(Nested, X);
  }

  while (nxy > bch_product_threshold)
  {
    if ((k < 2) or (k % 2 == 0)){
      Z += (bernoulli[k] / factorial[k]) * Nested;
    }
    k++;
    if (k >= bernoulli.size()) break; // don't evaluate the commutator if we're not going to use it
    if (2 * ny * nxy < bch_product_threshold) break;

    Nested = Commutator::Commutator(Y, Nested);
    nxy = Nested.Norm();
  }

  Profiler::timer[__func__] += omp_get_wtime() - start_time;
  return Z;
}

void IMSRG::WriteFlowStatus(std::ostream &f)
{
  if (f.good())
  {
    int fwidth = 16;
    int fprecision = 6;
    f.setf(std::ios::fixed);
    f << std::fixed << std::setw(5) << istep
      << std::setw(12) << std::setprecision(5) << s
      << std::setw(fwidth) << std::setprecision(fprecision) << std::real(Hs.ZeroBody) / (double)Hs.modelspace->A
      << std::scientific << std::setw(fwidth) << std::setprecision(fprecision) << Hs.Norm()
//      << std::setw(fwidth) << std::setprecision(fprecision) << Omega_s.back().OneBody.Norm()
      << std::setw(fwidth) << std::setprecision(fprecision) << Omega_s.back().TwoBody.Norm()
//      << std::setw(fwidth) << std::setprecision(fprecision) << Eta.OneBody.Norm()
      << std::setw(fwidth) << std::setprecision(fprecision) << Eta.TwoBody.Norm()
      << std::setw(7) << std::setprecision(0) << Profiler::counter["N_Commutators"]
      << std::setw(7) << std::setprecision(0) << Profiler::counter["N_Operators"]
      << std::setw(16) << std::fixed << std::setprecision(2) << Profiler::GetTimes()["walltime"]
      << std::setw(10) << std::setprecision(3) << Profiler::CheckMem()["RSS"] / 1024.0 / 1024.0 << " / " << std::skipws << Profiler::MaxMemUsage() / 1024.0 / 1024.0
      << std::endl;
  }
}

void IMSRG::WriteFlowStatusHeader(std::ostream &f)
{
  if (f.good())
  {
    int fwidth = 16;
    int fprecision = 9;
    f.setf(std::ios::fixed);
    f << std::fixed << std::setw(5) << "i"
      << std::setw(12) << std::setprecision(5) << "s"
      << std::setw(fwidth) << std::setprecision(fprecision) << "E0/A"
      << std::setw(fwidth) << std::setprecision(fprecision) << "||H||"
//      << std::setw(fwidth) << std::setprecision(fprecision) << "||Omega_1||"
      << std::setw(fwidth) << std::setprecision(fprecision) << "||Omega_2||"
//      << std::setw(fwidth) << std::setprecision(fprecision) << "||Eta_1||"
      << std::setw(fwidth) << std::setprecision(fprecision) << "||Eta_2||"
      << std::setw(7) << std::setprecision(fprecision) << "Ncomm"
      << std::setw(7) << std::setprecision(fprecision) << "Nops"
      << std::setw(16) << std::setprecision(fprecision) << "Walltime (s)"
      << std::setw(18) << std::setprecision(fprecision) << "Memory (GB)"
      << std::endl;
    for (int x = 0; x < 140; x++)
      f << "-";
    f << std::endl;
  }
}

void IMSRG::solve_magnus()
{
  istep = 0;
  if(s < 1.e-4) WriteFlowStatusHeader(std::cout);
  Eta.SetGenerator(Hs);
  WriteFlowStatus(std::cout);

  for(istep = 1; s < smax; istep++){
    double norm_eta = Eta.Norm();
    if (norm_eta < eta_criterion) break;
    double norm_omega = Omega_s.back().Norm();
    if(norm_omega > omega_norm_max) NewOmega();

    ds = std::min(ds_max, smax - s);
    s += ds;
    Eta *= ds;
    Omega_s.back() = BCHProduct(Eta, Omega_s.back());
    if(Omega_s.size() < 2) {
      Hs = BCHTransformation(*Hin, Omega_s.back());
    }
    else {
      Hs = BCHTransformation(H_saved, Omega_s.back());
    }
    Eta.SetGenerator(Hs);
    WriteFlowStatus(std::cout);
  }
}

void IMSRG::solve_runge_kutta4()
{
  istep = 0;
  Eta.SetGenerator(Hs);
  WriteFlowStatusHeader(std::cout);
  WriteFlowStatus(std::cout);
  for(istep = 1; s < smax; istep++){
    double norm_eta = Eta.Norm();
    if (norm_eta < eta_criterion) break;
    ds = std::min(ds_max, smax - s);
    s += ds;

    Operator tmp1 = Commutator::Commutator(Eta, Hs);
    Operator tmp = Hs + 0.5 * ds * tmp1;
    Eta.SetGenerator(tmp);

    Operator tmp2 = Commutator::Commutator(Eta, tmp);
    tmp = Hs + 0.5 * ds * tmp2;
    Eta.SetGenerator(tmp);

    Operator tmp3 = Commutator::Commutator(Eta, tmp);
    tmp = Hs + ds * tmp3;
    Eta.SetGenerator(tmp);

    Operator tmp4 = Commutator::Commutator(Eta, tmp);
    Hs += (ds / 6.0) * (tmp1 + 2.0*tmp2 + 2.0*tmp3 + tmp4);
    Eta.SetGenerator(Hs);
    WriteFlowStatus(std::cout);
  }
}

void IMSRG::NewOmega()
{
  H_saved = Hs;
  std::cout << "pushing back another Omega. Omega_s.size = " << Omega_s.size()
            << " , operator size = " << Omega_s.front().Memory() / (1024.0 * 1024.0 * 1024.0)  << " GB";
  std::cout << std::endl;
  Omega_s.emplace_back(Eta);
  Omega_s.back().Erase();
  Omega_s.back().SetAntiSym(true);
}

