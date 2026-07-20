#include "MBPT.hh"
#include "omp_complex_reduction.hh"

//MBPT::~MBPT(){}
//
//MBPT::MBPT(const Operator& Ham):
//  Ham(&Ham)
//{}

std::complex<double> MBPT::EnergyCorrection2(const Operator& Ham)
{
  double start_time = omp_get_wtime();
  std::complex<double> res = {0.0,0.0};
#pragma omp parallel for schedule(dynamic) reduction(+: res)
  for(int ich=0; ich<Ham.modelspace->TwoBody.GetNumberChannels(); ich++){
    TwoBodyChannel& tbc = Ham.modelspace->TwoBody.GetChannel(ich);
    for(int idx_hh : tbc.index_hh){
      int ii = tbc.GetSPStateIndex1(idx_hh);
      int ij = tbc.GetSPStateIndex2(idx_hh);
      SPState& i = Ham.modelspace->GetSPState(ii);
      SPState& j = Ham.modelspace->GetSPState(ij);
      for(int idx_pp : tbc.index_pp){
        int ia = tbc.GetSPStateIndex1(idx_pp);
        int ib = tbc.GetSPStateIndex2(idx_pp);
        SPState& a = Ham.modelspace->GetSPState(ia);
        SPState& b = Ham.modelspace->GetSPState(ib);
        std::complex<double> denom = Ham.Get1BME(ii,ii) + Ham.Get1BME(ij,ij) - Ham.Get1BME(ia,ia) - Ham.Get1BME(ib,ib);
        std::complex<double> me = std::norm(Ham.Get2BME(ii,ij,ia,ib));
        res += i.occ * j.occ * (1.0-a.occ) * (1.0-b.occ) * me / denom;
      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
  return res;
}
