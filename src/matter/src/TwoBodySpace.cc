#include "TwoBodySpace.hh"

TwoBodySpace::~TwoBodySpace()
{}

TwoBodySpace::TwoBodySpace()
{}

TwoBodySpace::TwoBodySpace(std::vector<SPState>& SPStates):
  SPStates(&SPStates)
{
  int Nxmin, Nxmax, Nymin, Nymax, Nzmin, Nzmax;
  Nxmin = 9999; Nxmax = -9999; Nymin = 9999; Nymax = -9999; Nzmin = 9999; Nzmax = -9999;
  for(auto & p : SPStates){
    for(auto & q : SPStates){
      Nxmin = std::min(Nxmin, p.nx + q.nx);
      Nxmax = std::max(Nxmax, p.nx + q.nx);
      Nymin = std::min(Nymin, p.ny + q.ny);
      Nymax = std::max(Nymax, p.ny + q.ny);
      Nzmin = std::min(Nzmin, p.nz + q.nz);
      Nzmax = std::max(Nzmax, p.nz + q.nz);
    }
  }

  std::vector<std::array<int,4>> tmp;
  for(int Tz : {-1, 0, 1}){
    for(int Nx=Nxmin; Nx<=Nxmax; Nx++){
      for(int Ny=Nymin; Ny<=Nymax; Ny++){
        for(int Nz=Nzmin; Nz<=Nzmax; Nz++){
          tmp.push_back({Tz,Nx,Ny,Nz});
        }
      }
    }
  }

  // Normal channels
  int ChIndex = 0;
  std::vector<TwoBodyChannel> local_channels;
  std::map<std::array<int, 4>, int> local_channel_index;
  int local_ch_index = ChIndex;  // Thread-local ChIndex
#pragma omp for
  for (auto &it : tmp) {
    int Tz = it[0];
    int Nx = it[1];
    int Ny = it[2];
    int Nz = it[3];
    TwoBodyChannel tmp = TwoBodyChannel(SPStates, Tz, Nx, Ny, Nz, local_ch_index);

    if (tmp.GetNumberStates() == 0) {
      local_channel_index[{Tz, Nx, Ny, Nz}] = -9999;
      continue;
    }

    // Collect results in thread-local storage
    local_channels.emplace_back(SPStates, Tz, Nx, Ny, Nz, local_ch_index);
    local_channel_index[{Tz, Nx, Ny, Nz}] = local_ch_index;
    local_ch_index += 1;
  }

  // Combine thread-local results into the global shared data outside the parallel loop
#pragma omp critical
  {
    Channels.insert(Channels.end(), local_channels.begin(), local_channels.end());
    ChannelIndex.insert(local_channel_index.begin(), local_channel_index.end());
    ChIndex = local_ch_index; // Update global ChIndex
  }
  NumberChannels = Channels.size();

  // Cross coupled channels
  int XChIndex = 0;
  std::vector<XTwoBodyChannel> local_xchannels;
  std::map<std::array<int, 4>, int> local_xchannel_index;
  int local_xch_index = XChIndex;  // Thread-local ChIndex
#pragma omp for
  for (auto &it : tmp) {
    int Tz = it[0];
    int Nx = it[1];
    int Ny = it[2];
    int Nz = it[3];
    XTwoBodyChannel tmp = XTwoBodyChannel(SPStates, Tz, Nx, Ny, Nz, local_xch_index);

    if (tmp.GetNumberStates() == 0) {
      local_xchannel_index[{Tz, Nx, Ny, Nz}] = -9999;
      continue;
    }

    // Collect results in thread-local storage
    local_xchannels.emplace_back(SPStates, Tz, Nx, Ny, Nz, local_xch_index);
    local_xchannel_index[{Tz, Nx, Ny, Nz}] = local_xch_index;
    local_xch_index += 1;
  }

  // Combine thread-local results into the global shared data outside the parallel loop
#pragma omp critical
  {
    XChannels.insert(XChannels.end(), local_xchannels.begin(), local_xchannels.end());
    XChannelIndex.insert(local_xchannel_index.begin(), local_xchannel_index.end());
    XChIndex = local_xch_index; // Update global ChIndex
  }
  NumberXChannels = XChannels.size();

  // Sorting
  std::sort(Channels.begin(), Channels.end(), [](const TwoBodyChannel& a, const TwoBodyChannel& b) {
      return a.GetNumberStates() > b.GetNumberStates();
      });

  std::sort(XChannels.begin(), XChannels.end(), [](const XTwoBodyChannel& a, const XTwoBodyChannel& b) {
      return a.GetNumberStates() > b.GetNumberStates();
      });

  int counter = 0;
  int idx = 0;
  for(auto& channel : Channels){
    ChannelIndex[{channel.Tz, channel.Nx, channel.Ny, channel.Nz}] = idx;
    channel.ChannelIndex = idx;
    idx += 1;
    counter += channel.GetNumberStates();
  }

  idx = 0;
  for(auto& channel : XChannels){
    XChannelIndex[{channel.Tz, channel.nx, channel.ny, channel.nz}] = idx;
    channel.ChannelIndex = idx;
    idx += 1;
    counter += channel.GetNumberStates();
  }
  std::cout << "Max size of two-body channel: " << Channels[0].GetNumberStates() << std::endl;
  std::cout << "Max size of cross-coupled two-body channel: " << XChannels[0].GetNumberStates() << std::endl;
}


int TwoBodySpace::GetChannelIndex(int ip, int iq)
{
  SPState& p = (*SPStates)[ip];
  SPState& q = (*SPStates)[iq];
  return GetChannelIndex((p.tz+q.tz)/2, p.nx+q.nx, p.ny+q.ny, p.nz+q.nz);
}

int TwoBodySpace::GetXChannelIndex(int ip, int iq)
{
  SPState& p = (*SPStates)[ip];
  SPState& q = (*SPStates)[iq];
  return GetXChannelIndex((p.tz-q.tz)/2, p.nx-q.nx, p.ny-q.ny, p.nz-q.nz);
}
