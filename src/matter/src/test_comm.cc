#include "Parameters.hh"
#include "Constants.hh"
#include "SPState.hh"
#include "ModelSpace.hh"
#include "Operator.hh"
#include "MBPT.hh"
#include "Commutator.hh"

int main(int argc, char** argv)
{
  Profiler::Start();
  Parameters parameters(argc,argv);
  std::string mode = parameters.s("mode");
  bool chiral_delta = (parameters.s("chiral_delta")=="true") ? true : false;
  bool include_3NF = (parameters.s("include_3NF")=="true") ? true : false;
  int A = parameters.i("A");
  int Nmax = parameters.i("Nmax");
  int ch_order = parameters.i("ch_order");
  int reg_power = parameters.i("reg_power");

  double rho = parameters.d("rho");
  double RegLambda = parameters.d("RegLambda");
  double SFRLambda = parameters.d("SFRLambda");
  std::map<std::string, double> LECs;
  LECs["Ct1S0_pp"] = parameters.d("Ct1S0_pp");
  LECs["Ct1S0_np"] = parameters.d("Ct1S0_np");
  LECs["Ct1S0_nn"] = parameters.d("Ct1S0_nn");
  LECs["Ct3S1"] = parameters.d("Ct3S1");
  LECs["C1S0"] = parameters.d("C1S0");
  LECs["C3S1"] = parameters.d("C3S1");
  LECs["C1P1"] = parameters.d("C1P1");
  LECs["C3P0"] = parameters.d("C3P0");
  LECs["C3P1"] = parameters.d("C3P1");
  LECs["C3SD1"] = parameters.d("C3SD1");
  LECs["C3P2"] = parameters.d("C3P2");
  LECs["c1"] = parameters.d("c1");
  LECs["c2"] = parameters.d("c2");
  LECs["c3"] = parameters.d("c3");
  LECs["c4"] = parameters.d("c4");
  LECs["cD"] = parameters.d("cD");
  LECs["cE"] = parameters.d("cE");

  std::cout << "# Number of threads: " << std::setw(4) << omp_get_max_threads() << std::endl;
  ModelSpace modelspace = ModelSpace(A, Nmax, rho, "box", mode, {0.0, 0.0, 0.0}, true);

  Operator Ham = Operator(modelspace);
  Ham.OneBody.SetKineticEnergy();
  Ham.TwoBody.SetNNInteraction(ch_order, LECs, RegLambda, reg_power, SFRLambda, chiral_delta);
  Ham.AllocateV3NO2B();
  if(include_3NF) Ham.V3NO2B.SetNO2B3NInteraction(ch_order, LECs, RegLambda, reg_power, SFRLambda, chiral_delta);

  Ham.NormalOrderingH();
  std::cout << __LINE__ << std::endl;
  Ham.PrintEnergy();
  MBPT mbpt = MBPT();
  std::complex<double> E2 = mbpt.EnergyCorrection2(Ham);
  std::cout << "MBPT2 correction (E2/A): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(E2) / (double)modelspace.A
    << " + " << std::scientific << std::setw(12) << std::setprecision(4) << std::imag(E2) / (double)modelspace.A << "i MeV" << std::endl;

  Ham.PrintSPE();
  Operator Eta = Operator(modelspace);
  Eta.SetAntiSym(true);
  Eta.SetGenerator(Ham);

  Operator comm = Commutator::Commutator(Eta, Ham);
  std::cout << "fast implementation 0B: " << std::real(comm.ZeroBody) << ", 1B norm: "  << std::abs(comm.OneBody.Norm()) << ", 2B norm: " << std::abs(comm.TwoBody.Norm()) << std::endl;;

  Commutator::naive_implementation(true);
  Operator comm_naive = Commutator::Commutator(Eta, Ham);
  std::cout << "slow implementation 0B: " << std::real(comm_naive.ZeroBody) << ", 1B norm: "  << std::abs(comm_naive.OneBody.Norm()) << ", 2B norm: " << std::abs(comm_naive.TwoBody.Norm()) << std::endl;;
  comm -= comm_naive;
  std::cout << "diff 0B: " << std::real(comm.ZeroBody) << ", 1B norm: "  << std::abs(comm.OneBody.Norm()) << ", 2B norm: " << std::abs(comm.TwoBody.Norm()) << std::endl;

  Profiler::PrintTimes();
}

