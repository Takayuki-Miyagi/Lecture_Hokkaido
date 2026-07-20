#include "TwoBodyChannel.hh"

TwoBodyChannel::~TwoBodyChannel()
{}

TwoBodyChannel::TwoBodyChannel(std::vector<SPState>& SPStates, int Tz, int Nx, int Ny, int Nz, int ChannelIndex):
  SPStates(&SPStates), Tz(Tz), Nx(Nx), Ny(Ny), Nz(Nz), ChannelIndex(ChannelIndex)
{
  int n = 0;
  for(int ip = 0; ip < (int)SPStates.size(); ip++){
    for(int iq = 0; iq < ip; iq++){
      SPState& p = SPStates[ip];
      SPState& q = SPStates[iq];
      if((p.tz + q.tz)/2 != Tz) continue;
      if((p.nx + q.nx) != Nx) continue;
      if((p.ny + q.ny) != Ny) continue;
      if((p.nz + q.nz) != Nz) continue;
      index_to_pq.push_back({p.index, q.index});
      index[{p.index, q.index}] = n;
      index[{q.index, p.index}] = n;
      phase[{p.index, q.index}] = 1;
      phase[{q.index, p.index}] =-1;
      n += 1;
    }
  }
  NumberStates = n;
  for(int index=0; index<NumberStates; index++){
    int ip = GetSPStateIndex1(index);
    int iq = GetSPStateIndex2(index);
    SPState& p = SPStates[ip];
    SPState& q = SPStates[iq];
    if(p.occ + q.occ < 1.e-4) index_pp.push_back(index);
    if(p.occ + q.occ > 1.e-4 and p.occ * q.occ < 1.e-4) index_ph.push_back(index);
    if(p.occ + q.occ > 1.e-4 and p.occ * q.occ > 1.e-4) index_hh.push_back(index);
    if(std::abs(1.0 - p.occ - q.occ) > 1.e-4) {
      index_for_comm_pphh.push_back(index);
      factor_for_comm_pphh.push_back(1.0-p.occ-q.occ);
    }
  }
}

XTwoBodyChannel::~XTwoBodyChannel()
{}

XTwoBodyChannel::XTwoBodyChannel(std::vector<SPState>& SPStates, int Tz, int nx, int ny, int nz, int ChannelIndex):
  SPStates(&SPStates), Tz(Tz), nx(nx), ny(ny), nz(nz), ChannelIndex(ChannelIndex)
{
  int n = 0;
  for(int ip = 0; ip < (int)SPStates.size(); ip++){
    for(int iq = 0; iq < (int)SPStates.size(); iq++){
      SPState& p = SPStates[ip];
      SPState& q = SPStates[iq];
      if((p.tz - q.tz)/2 != Tz) continue;
      if((p.nx - q.nx) != nx) continue;
      if((p.ny - q.ny) != ny) continue;
      if((p.nz - q.nz) != nz) continue;
      index_to_pq.push_back({p.index, q.index});
      index[{p.index, q.index}] = n;
      n += 1;
    }
  }
  NumberStates = n;
  for(int index=0; index<NumberStates; index++){
    int ip = GetSPStateIndex1(index);
    int iq = GetSPStateIndex2(index);
    SPState& p = SPStates[ip];
    SPState& q = SPStates[iq];
    if(p.occ + q.occ < 1.e-4) index_pp.push_back(index);
    if(p.occ + q.occ > 1.e-4 and p.occ * q.occ < 1.e-4) index_ph.push_back(index);
    if(p.occ + q.occ > 1.e-4 and p.occ * q.occ > 1.e-4) index_hh.push_back(index);
    if(std::abs(p.occ - q.occ) > 1.e-4) {
      index_for_comm_ph.push_back(index);
      factor_for_comm_ph.push_back(p.occ-q.occ);
    }
  }
}
