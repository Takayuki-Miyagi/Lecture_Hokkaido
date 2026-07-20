#include <gsl/gsl_integration.h>
#include "Parameters.hh"
#include "Profiler.hh"
#include "Constants.hh"
#include "SPState.hh"
#include "ModelSpace.hh"
#include "Operator.hh"
#include "MBPT.hh"
#include "IMSRG.hh"


int main(int argc, char** argv)
{
  Profiler::Start();
  Parameters parameters(argc,argv);
  std::string mode = parameters.s("mode");
  std::string method = parameters.s("method");
  std::string input_nn_file = parameters.s("input_nn_file");
  std::string truncation_method = parameters.s("truncation_method");
  bool chiral_delta = (parameters.s("chiral_delta")=="true") ? true : false;
  bool include_3NF = (parameters.s("include_3NF")=="true") ? true : false;
  int A = parameters.i("A");
  int Zu = parameters.i("Zu");
  int Nu = parameters.i("Nu");
  int Zd = parameters.i("Zd");
  int Nd = parameters.i("Nd");
  int Nmax = parameters.i("Nmax");
  int ch_order = parameters.i("ch_order");
  int reg_power = parameters.i("reg_power");
  int n_TABC = parameters.i("n_TABC");

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

  double smax = parameters.d("smax");
  double ds_max = parameters.d("ds_max");
  double omega_norm_max = parameters.d("omega_norm_max");
  double eta_criterion = parameters.d("eta_criterion");

  std::cout << "# Number of threads: " << std::setw(4) << omp_get_max_threads() << std::endl;

  std::vector<double> twist_angles, weights;
  if(n_TABC==-1) {
    std::cout << std::endl;
    std::cout << "Periodic boundary condition will be used" << std::endl;
    std::cout << "Truncation method: " << truncation_method << std::endl;
    std::cout << std::endl;
    twist_angles.push_back(0.0);
    weights.push_back(1.0);
  }
  else{
    std::cout << "Don't use this option!" << std::endl;
    exit(0);
    std::cout << std::endl;
    std::cout << "Twist-averaged boundary condition will be used. Number of points for each direction is " << n_TABC << std::endl;
    std::cout << std::endl;
    gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(n_TABC);
    for (int i=0; i<n_TABC; i++){
      double x, w;
      gsl_integration_glfixed_point(0.0, Constants::PI, i, &x, &w, table);
      twist_angles.push_back(x);
      weights.push_back(w / Constants::PI); // 1/pi is from that we want to compute the average
    }
    gsl_integration_glfixed_table_free(table);
  }

  std::vector<std::array<int,3>> TABC_loop;
  for(int ix=0; ix<twist_angles.size(); ix++){
    for(int iy=0; iy<twist_angles.size(); iy++){
      for(int iz=0; iz<twist_angles.size(); iz++){
        TABC_loop.push_back({ix,iy,iz});
      }
    }
  }

  std::vector<std::vector<std::complex<double>>> results;
  std::complex<double> ehf = {0.0,0.0};
  for(int loop_index=0; loop_index<TABC_loop.size(); loop_index++){
    std::cout << "Looping " << loop_index+1 << "/" << TABC_loop.size() << std::endl;
    std::cout << std::endl;
    double thetax = twist_angles[TABC_loop[loop_index][0]];
    double thetay = twist_angles[TABC_loop[loop_index][1]];
    double thetaz = twist_angles[TABC_loop[loop_index][2]];
    double integral_weight = weights[TABC_loop[loop_index][0]] * weights[TABC_loop[loop_index][1]] * weights[TABC_loop[loop_index][2]];
    std::vector<std::complex<double>> res;
    bool verbose = true;
    if(loop_index>0) verbose = false;

    ModelSpace modelspace = (mode=="none") ? ModelSpace(Zu, Zd, Nu, Nd, Nmax, rho, truncation_method, {thetax, thetay, thetaz}, verbose) :
      ModelSpace(A, Nmax, rho, truncation_method, mode, {thetax, thetay, thetaz}, verbose);
    Operator Ham = Operator(modelspace);
    if(verbose) std::cout << "Hamiltonian is using" << std::fixed << std::setw(12) << std::setprecision(4) << Ham.Memory() / (1024.0 * 1024.0 * 1024.0) << " GB memory" << std::endl;

    Ham.OneBody.SetKineticEnergy();
    if(input_nn_file == "none"){
      Ham.TwoBody.SetNNInteraction(ch_order, LECs, RegLambda, reg_power, SFRLambda, chiral_delta);
    }
    else{
      Ham.TwoBody.SetNNInteraction(input_nn_file);
    }
    Ham.AllocateV3NO2B();
    if(include_3NF) Ham.V3NO2B.SetNO2B3NInteraction(ch_order, LECs, RegLambda, reg_power, SFRLambda, chiral_delta);

    Ham.NormalOrderingH();
    Ham.PrintEnergy();
    if(verbose) Ham.PrintSPE();
    res.push_back(Ham.ZeroBody*integral_weight);
    ehf += Ham.ZeroBody*integral_weight;

    if(method=="HF") {
      continue;
    }

    MBPT mbpt = MBPT();
    std::complex<double> E2 = mbpt.EnergyCorrection2(Ham);
    std::cout << "MBPT2 Enregy (E/A): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(Ham.ZeroBody+E2) / (double)modelspace.A
      << " + " << std::scientific << std::setw(12) << std::setprecision(4) << std::imag(Ham.ZeroBody+E2) / (double)modelspace.A << "i MeV" << std::endl;
    std::cout << std::endl;
    res.push_back((Ham.ZeroBody+E2)*integral_weight);

    if(method=="MBPT") {
      continue;
    }

    IMSRG imsrg = IMSRG(Ham, smax, ds_max, omega_norm_max, eta_criterion);
    imsrg.Solve(method);
    Ham = imsrg.Hs;
    std::cout << std::endl;
    std::cout << "IMSRG Energy (E/A): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(Ham.ZeroBody) / (double)modelspace.A
      << " + " << std::scientific << std::setw(12) << std::setprecision(4) << std::imag(Ham.ZeroBody) / (double)modelspace.A << "i MeV" << std::endl;
    std::cout << std::endl;
    res.push_back(Ham.ZeroBody*integral_weight);

    if(method=="RK4") {
      continue;
    }

    std::cout << std::endl;
    //  for (std::string opname : {"Spin", "T", "NN", "3N"}){
    //
    //    Operator op = Operator(modelspace);
    //    if(opname=="Spin") {
    //      std::cout << "Spin operator (Sz): " << std::endl;
    //      op.SetSpinOperator();
    //      op.NormalOrdering();
    //      std::cout << "<Sz>/A (HF):    " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //      op = imsrg.Evolve(op);
    //      std::cout << "<Sz>/A (IMSRG): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //    }
    //
    //    if(opname=="T") {
    //      std::cout << "Kinetic Energy: " << std::endl;
    //      op.OneBody.SetKineticEnergy();
    //      op.NormalOrdering();
    //      std::cout << "<T>/A (HF):    " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //      op = imsrg.Evolve(op);
    //      std::cout << "<T>/A (IMSRG): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //    }
    //
    //    if(opname=="NN") {
    //      std::cout << "NN: " << std::endl;
    //      op.TwoBody.SetNNInteraction(ch_order, LECs, RegLambda, reg_power, SFRLambda, chiral_delta);
    //      op.NormalOrdering();
    //      std::cout << "<VNN>/A (HF):    " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //      op = imsrg.Evolve(op);
    //      std::cout << "<VNN>/A (IMSRG): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //    }
    //
    //    if(opname=="3N") {
    //      std::cout << "3N: " << std::endl;
    //      if(include_3NF) op.TwoBody.SetNO2B3NInteraction(ch_order, LECs, RegLambda, reg_power, SFRLambda, chiral_delta);
    //      op.NormalOrdering();
    //      op.OneBody *= 0.5;
    //      op.ZeroBody /= 3.0;
    //      std::cout << "<V3N>/A (HF):    " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //      op = imsrg.Evolve(op);
    //      std::cout << "<V3N>/A (IMSRG): " << std::fixed << std::setw(12) << std::setprecision(6) << std::real(op.ZeroBody)/(double)modelspace.A
    //        << " + " << std::imag(op.ZeroBody)/(double)modelspace.A << "i " << std::endl;
    //    }
    //    std::cout << std::endl;
    //  }
  }
  Profiler::PrintAll();
}
