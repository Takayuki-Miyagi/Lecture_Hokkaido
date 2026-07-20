#ifndef TwoBodySpace_h
#define TwoBodySpace_h 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include "SPState.hh"
#include "TwoBodyChannel.hh"

class TwoBodySpace
{
  public:
    int NumberChannels;
    int NumberXChannels;
    std::vector<TwoBodyChannel> Channels;
    std::vector<XTwoBodyChannel> XChannels;
    std::vector<SPState>* SPStates;
    std::map<std::array<int,4>,int> ChannelIndex;
    std::map<std::array<int,4>,int> XChannelIndex;

    ~TwoBodySpace();
    TwoBodySpace();
    TwoBodySpace(std::vector<SPState>& SPStates);
    int GetNumberChannels() const {return NumberChannels;};
    int GetNumberXChannels() const {return NumberXChannels;};
    int GetChannelIndex(int Tz, int Nx, int Ny, int Nz) const {return ChannelIndex.at({Tz,Nx,Ny,Nz});};
    TwoBodyChannel & GetChannel(int idx) {return Channels.at(idx);};
    int GetChannelIndex(int ip, int iq);

    int GetXChannelIndex(int Tz, int Nx, int Ny, int Nz) const {return XChannelIndex.at({Tz,Nx,Ny,Nz});};
    XTwoBodyChannel & GetXChannel(int idx) {return XChannels.at(idx);};
    int GetXChannelIndex(int ip, int iq);
};
#endif
