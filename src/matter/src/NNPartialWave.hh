#ifndef NNPartialWave_h
#define NNPartialWave_h 1
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>

class NNPartialWave
{
  public:
    int NMesh;
    int NChan;
    int Jmax;
    bool Weighted;
    std::string filename;
    std::vector<std::array<int,5>> JLLSTz;
    std::map<std::array<int,5>,int> ChannelIndex;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> Vmat;
    Eigen::Vector<double, Eigen::Dynamic> Mesh;
    Eigen::Vector<double, Eigen::Dynamic> Weight;

    ~NNPartialWave();
    NNPartialWave();
    NNPartialWave(std::string);
    int GetNumberChannels() {return JLLSTz.size();};
    int GetNumberMeshes() {return NMesh;};
    int GetJmax() {return Jmax;};
    int GetChannelIndex(int J, int L, int Lp, int S, int Tz){return ChannelIndex[{J,L,Lp,S,Tz}];};
    void SolveDeuteron();
    void WeightNormalize();
    void WeightUnNormalize();
    void FixPhase();
};
#endif
