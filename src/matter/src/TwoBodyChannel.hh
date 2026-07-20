#ifndef TwoBodyChannel_h
#define TwoBodyChannel_h 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <algorithm>
#include <map>
#include <Eigen/Dense>
#include "SPState.hh"

class TwoBodyChannel
{
  public:
    std::vector<SPState>* SPStates;
    int Tz;
    int Nx, Ny, Nz; // This is because that the cm momentum is conseved in initial and final states
    int ChannelIndex;
    int NumberStates;
    std::vector<std::array<int,2>> index_to_pq;               // index -> (p,q)
    std::map<std::array<int,2>,int> index; // (p,q) -> index
    std::map<std::array<int,2>,int> phase; // Antisymmetrization phase, should be 1 for (p,q), but -1 for (q,p)
    std::vector<int> index_hh, index_ph, index_pp, index_for_comm_pphh;
    std::vector<double> factor_for_comm_pphh;

    ~TwoBodyChannel();
    TwoBodyChannel(std::vector<SPState>& SPStates, int Tz, int Nx, int Ny, int Nz, int ChannelIndex);
    int GetTz(){return Tz;};
    const Eigen::Vector<double, 3> GetCMVector() {Eigen::Vector<double, 3> res; res << Nx, Ny, Nz; return res;};
    int GetNumberStates() const {return NumberStates;};
    int GetSPStateIndex1(int idx) const {return index_to_pq[idx][0];};
    int GetSPStateIndex2(int idx) const {return index_to_pq[idx][1];};
    int GetIndex(int p, int q) const {return index.at({p,q});};
    int GetPhase(int p, int q) const {return phase.at({p,q});};
    std::array<int,2> GetSPStateIndices(int idx) const {return index_to_pq[idx];};
};

class XTwoBodyChannel
{
  //
  // <pq|O|rs> --> <p \bar{s}|O|r \bar{q}>
  // kp + kq = kr + ks --> kp - ks = kr - kq
  //
  public:
    std::vector<SPState>* SPStates;
    int Tz;
    int nx, ny, nz; // k1 - k2 is conseved:
    int ChannelIndex;
    int NumberStates;
    std::vector<std::array<int,2>> index_to_pq; // index -> (p,q)
    std::map<std::array<int,2>,int> index; // (p,q) -> index
    std::vector<int> index_hh, index_ph, index_pp, index_for_comm_ph;
    std::vector<double> factor_for_comm_ph;

    ~XTwoBodyChannel();
    XTwoBodyChannel(std::vector<SPState>& SPStates, int Tz, int nx, int ny, int nz, int ChannelIndex);
    int GetTz() const {return Tz;};
    const Eigen::Vector<double, 3> GetRelVector() {Eigen::Vector<double, 3> res; res << nx, ny, nz; return res;};
    int GetNumberStates() const {return NumberStates;};
    int GetSPStateIndex1(int idx) const {return index_to_pq[idx][0];};
    int GetSPStateIndex2(int idx) const {return index_to_pq[idx][1];};
    int GetIndex(int p, int q) const {return index.at({p,q});};
    std::array<int,2> GetSPStateIndices(int idx) const {return index_to_pq[idx];};
};
#endif
